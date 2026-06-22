"""
engine.py — low-level driver for the headless Slay the Spire 2 engine.

This is the plumbing layer. It knows how to:
  1. launch the C# engine as a subprocess,
  2. send it JSON commands and read JSON replies,
  3. reset to a *fixed* combat (the scenario our agent will learn to win).

It does NOT know anything about reinforcement learning, neural networks, or
observations/rewards. Keeping that separation makes each layer easy to reason
about: engine.py = "talk to the game", sts_env.py (next file) = "make it an RL env".
"""
from __future__ import annotations

import json
import os
import subprocess

# Repo root is one level up from this file (…/sts2-cli/rl/engine.py -> …/sts2-cli)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB_DIR = os.path.join(REPO_ROOT, "lib")
# Launch the *compiled* DLL directly rather than `dotnet run`. `dotnet run` re-evaluates
# the project via MSBuild on every launch (~0.3s + lock contention under parallelism);
# the built DLL boots in ~0.05s. Build it once: `dotnet build src/Sts2Headless/...`.
ENGINE_DLL = os.path.join(
    REPO_ROOT, "src", "Sts2Headless", "bin", "Debug", "net9.0", "Sts2Headless.dll"
)

# ── The fixed scenario for Phase 1 ───────────────────────────────────────────
# A deterministic, winnable fight so the agent has a stable target to learn.
#   - Ironclad with the standard starter deck (we set it explicitly so a Neow
#     blessing can't change it between episodes).
#   - A single weak enemy.
# Card/encounter IDs are the game's internal model ids (no "CARD."/"ENCOUNTER." prefix here;
# the engine adds the category).
FIXED_CHARACTER = "Ironclad"
FIXED_SEED = "rl_fixed_combat"
FIXED_DECK = (
    ["STRIKE_IRONCLAD"] * 5
    + ["DEFEND_IRONCLAD"] * 4
    + ["BASH"]
)
FIXED_ENCOUNTER = "SHRINKER_BEETLE_WEAK"

# Decisions that can appear between starting a run and entering our combat
# (e.g. the Neow blessing screen). We resolve them with a trivial default so we
# always land in the same place.
_INTERMEDIATE = {"event_choice", "bundle_select", "card_select", "card_reward"}


class Engine:
    """Owns one engine subprocess and the JSON conversation with it."""

    def __init__(self) -> None:
        if not os.path.isfile(ENGINE_DLL):
            raise FileNotFoundError(
                f"Engine not built: {ENGINE_DLL}\n"
                f"Build it first: dotnet build src/Sts2Headless/Sts2Headless.csproj"
            )
        env = dict(os.environ)
        # Tell the engine where the patched game DLLs live (the lib/ folder).
        env["STS2_GAME_DIR"] = LIB_DIR
        self.proc = subprocess.Popen(
            ["dotnet", ENGINE_DLL],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,  # the engine prints lots of debug to stderr; ignore it
            text=True,
            cwd=REPO_ROOT,
            env=env,
        )
        # The engine emits one line, {"type":"ready",...}, when it has booted.
        self.last = self._read()
        self._started = False   # has start_run been done in this process yet?

    # --- raw IO -------------------------------------------------------------
    def _read(self) -> dict:
        """Read one JSON line of output from the engine."""
        while True:
            line = self.proc.stdout.readline()
            if not line:
                return {"type": "error", "message": "engine process ended (EOF)"}
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)

    def send(self, cmd: dict) -> dict:
        """Send one command dict and return the engine's JSON reply."""
        self.proc.stdin.write(json.dumps(cmd) + "\n")
        self.proc.stdin.flush()
        self.last = self._read()
        return self.last

    # --- convenience --------------------------------------------------------
    def action(self, name: str, **args) -> dict:
        """Shorthand for the common {"cmd":"action","action":name,"args":{...}}."""
        cmd = {"cmd": "action", "action": name}
        if args:
            cmd["args"] = args
        return self.send(cmd)

    def reset_to_fixed_combat(self) -> dict:
        """
        Return a fresh fixed-combat 'combat_play' state.

        The first call does the expensive one-time setup (start_run + ModelDb init,
        ~0.8s). Every call after that is an in-process *combat* reset (~1-2ms): we
        leave the previous fight, restore HP + deck, and drop into a new identical
        combat — no new process, no game re-init. This is the ~400x speedup that
        makes training practical. Falls back to a full process respawn if the fast
        path ever lands somewhere unexpected (engine crash, surprise decision).
        """
        if not self._started:
            return self._full_reset()
        try:
            return self._fast_reset()
        except Exception:
            self._respawn()
            return self._full_reset()

    def _full_reset(self) -> dict:
        """One-time path: start a run, clear Neow, set the deck, enter combat."""
        state = self.send({
            "cmd": "start_run",
            "character": FIXED_CHARACTER,
            "seed": FIXED_SEED,
            "ascension": 0,
        })
        for _ in range(20):  # clear pre-combat screens (Neow, etc.)
            if state.get("decision") not in _INTERMEDIATE:
                break
            state = self._resolve_intermediate(state)
        self._started = True
        return self._enter_fixed_combat()

    def _fast_reset(self) -> dict:
        """In-process path: drain the post-combat screen, then re-enter combat."""
        # After a fight ends we're on card_reward (win) or game_over (loss). Skip any
        # reward/selection screens so the run is in a clean state to enter a new room.
        state = self.last
        for _ in range(20):
            dec = state.get("decision")
            if dec not in _INTERMEDIATE:
                break  # map_select / game_over / etc. — safe to enter a room
            state = self._resolve_intermediate(state)
        state = self._enter_fixed_combat()
        if state.get("decision") != "combat_play":
            raise RuntimeError(f"fast reset landed on {state.get('decision')}")
        return state

    def _enter_fixed_combat(self) -> dict:
        """Restore HP + fixed deck, then start the fixed encounter."""
        self.send({"cmd": "set_player", "hp": 80, "deck": list(FIXED_DECK)})
        return self.send({"cmd": "enter_room", "type": "combat", "encounter": FIXED_ENCOUNTER})

    def _resolve_intermediate(self, state: dict) -> dict:
        """Answer one Neow/reward/selection screen with a trivial default."""
        dec = state.get("decision")
        if dec == "event_choice":
            return self.action("choose_option", option_index=0)
        if dec == "bundle_select":
            return self.action("select_bundle", bundle_index=0)
        if dec == "card_select":
            return self.action("select_cards", indices="0")
        if dec == "card_reward":
            return self.action("skip_card_reward")
        return state

    def _respawn(self) -> None:
        """Kill and relaunch the engine process (fallback for a bad state)."""
        try:
            self.proc.kill()
        except Exception:
            pass
        self.__init__()  # re-boot a fresh process

    def close(self) -> None:
        try:
            self.send({"cmd": "quit"})
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


# Manual smoke test: `python rl/engine.py`
if __name__ == "__main__":
    eng = Engine()
    s = eng.reset_to_fixed_combat()
    p = s.get("player", {})
    enemies = s.get("enemies") or []
    hand = s.get("hand") or []
    print("decision:", s.get("decision"))
    print(f"player: HP {p.get('hp')}/{p.get('max_hp')}  block {p.get('block')}  energy {s.get('energy')}")
    print("enemies:", [(e.get("name"), e.get("hp")) for e in enemies])
    print("hand:", [c.get("name") for c in hand])
    print("each hand card's legality + target type:")
    for i, c in enumerate(hand):
        print(f"  [{i}] {c.get('name'):10} can_play={c.get('can_play')} target={c.get('target_type')}")
    eng.close()
