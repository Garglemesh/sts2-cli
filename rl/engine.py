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
CSPROJ = os.path.join(REPO_ROOT, "src", "Sts2Headless", "Sts2Headless.csproj")
LIB_DIR = os.path.join(REPO_ROOT, "lib")

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
        env = dict(os.environ)
        # Tell the engine where the patched game DLLs live (the lib/ folder).
        env["STS2_GAME_DIR"] = LIB_DIR
        self.proc = subprocess.Popen(
            ["dotnet", "run", "--no-build", "--project", CSPROJ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,  # the engine prints lots of debug to stderr; ignore it
            text=True,
            cwd=REPO_ROOT,
        )
        # The engine emits one line, {"type":"ready",...}, when it has booted.
        self._read()

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
        return self._read()

    # --- convenience --------------------------------------------------------
    def action(self, name: str, **args) -> dict:
        """Shorthand for the common {"cmd":"action","action":name,"args":{...}}."""
        cmd = {"cmd": "action", "action": name}
        if args:
            cmd["args"] = args
        return self.send(cmd)

    def reset_to_fixed_combat(self) -> dict:
        """
        Start a fresh run and drive it to the start of our fixed combat.
        Returns the engine's 'combat_play' state dict (HP, hand, enemies, …).
        """
        state = self.send({
            "cmd": "start_run",
            "character": FIXED_CHARACTER,
            "seed": FIXED_SEED,
            "ascension": 0,
        })
        # Clear any pre-combat decision screens (Neow, etc.) with simple defaults.
        for _ in range(20):
            dec = state.get("decision")
            if dec not in _INTERMEDIATE:
                break
            if dec == "event_choice":
                state = self.action("choose_option", option_index=0)
            elif dec == "bundle_select":
                state = self.action("select_bundle", bundle_index=0)
            elif dec == "card_select":
                state = self.action("select_cards", indices="0")
            elif dec == "card_reward":
                state = self.action("skip_card_reward")

        # Force a known deck, then drop straight into the chosen encounter.
        self.send({"cmd": "set_player", "deck": FIXED_DECK})
        state = self.send({"cmd": "enter_room", "type": "combat", "encounter": FIXED_ENCOUNTER})
        return state

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
