"""
sts_env.py — the reinforcement-learning *environment* for one fixed combat.

This wraps the low-level engine (engine.py) in the standard interface that every
RL library expects: Gymnasium's `Env`. An RL algorithm only ever sees three things
through this class:

    reset()  -> (observation, info)          # start a fresh combat
    step(a)  -> (observation, reward, terminated, truncated, info)
    action_masks() -> bool[]                 # which actions are legal right now

Everything game-specific — turning the JSON game state into numbers (the
"observation"), deciding which actions exist, computing the reward — lives here.

Phase 2 scope: Ironclad (fixed starter deck) vs. a RANDOM encounter from a pool with
1–3 enemies. Multiple enemies means the agent must choose *which* enemy to target,
so the action space is now (card slot x enemy target) plus end-turn.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from engine import Engine, ENCOUNTER_POOL


# ── Encoding constants ───────────────────────────────────────────────────────
# A tiny "vocabulary" of card identities the network can recognize. The fixed
# starter deck only uses three; UNK is a catch-all so the encoding won't crash if
# an unexpected card shows up. (We'll grow this as we generalize.)
CARD_VOCAB = ["STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "BASH", "UNK"]
CARD_INDEX = {name: i for i, name in enumerate(CARD_VOCAB)}
V = len(CARD_VOCAB)

MAX_HAND = 10          # hand slots the action space exposes
MAX_ENEMIES = 5        # enemy target slots the action space exposes
# Action layout: a < MAX_HAND*MAX_ENEMIES means "play hand card (a//ME) at enemy
# (a%ME)"; the final id is "end turn".
N_ACTIONS = MAX_HAND * MAX_ENEMIES + 1
END_TURN_ACTION = N_ACTIONS - 1

# Status effects ("powers") the agent should see — keyed on the engine's English
# power names (we run the engine in English). Amounts are normalized by ~5.
POWER_VOCAB = ["strength", "dexterity", "vulnerable", "weak", "frail", "shrink"]
POWER_INDEX = {name: i for i, name in enumerate(POWER_VOCAB)}
P = len(POWER_VOCAB)
NORM_POWER = 5.0

# Rough normalizers so all observation numbers land in a similar (~0..1) range,
# which helps neural nets learn. They don't need to be exact.
MAX_HP = 80.0
NORM_BLOCK = 40.0
NORM_ENERGY = 3.0
NORM_PILE = 10.0


def _card_base_id(card: dict) -> str:
    """'CARD.STRIKE_IRONCLAD' -> 'STRIKE_IRONCLAD' (strip the category prefix)."""
    cid = (card.get("id") or "").split(".")[-1]
    return cid if cid in CARD_INDEX else "UNK"


def _onehot_card(card: dict) -> np.ndarray:
    v = np.zeros(V, dtype=np.float32)
    v[CARD_INDEX[_card_base_id(card)]] = 1.0
    return v


def _pile_counts(cards: list[dict]) -> np.ndarray:
    """Count how many of each known card type are in a pile (order-independent)."""
    counts = np.zeros(V, dtype=np.float32)
    for c in cards or []:
        counts[CARD_INDEX[_card_base_id(c)]] += 1.0
    return counts / NORM_PILE


def _power_vec(powers: list[dict]) -> np.ndarray:
    """Encode a list of status effects as a normalized amount per known power."""
    vec = np.zeros(P, dtype=np.float32)
    for pw in powers or []:
        key = (pw.get("name") or "").lower()
        if key in POWER_INDEX:
            vec[POWER_INDEX[key]] = float(pw.get("amount", 0)) / NORM_POWER
    return vec


def _incoming_damage(enemy: dict) -> float:
    """Total damage this enemy intends to deal next turn (0 if it isn't attacking)."""
    dmg = 0.0
    for it in enemy.get("intents") or []:
        if it.get("total_damage") is not None:
            dmg += float(it["total_damage"])
        elif it.get("damage") is not None:
            dmg += float(it["damage"]) * float(it.get("hits") or 1)
    return dmg


def _intent_flags(enemy: dict) -> tuple[float, float]:
    """(is_attack, is_debuff) over the enemy's intents — what kind of move is coming."""
    types = {(it.get("type") or "") for it in (enemy.get("intents") or [])}
    is_attack = 1.0 if any("Attack" in t for t in types) else 0.0
    is_debuff = 1.0 if any("Debuff" in t for t in types) else 0.0
    return is_attack, is_debuff


class StsCombatEnv(gym.Env):
    """One fixed Slay the Spire 2 combat as a Gymnasium environment."""

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.engine: Engine | None = None
        self.state: dict = {}
        self._prev_hp: float = MAX_HP

        # --- Action space: (hand slot x enemy target) + end turn. ---
        self.action_space = spaces.Discrete(N_ACTIONS)

        # --- Observation space: a flat vector of floats. Layout (mirrors what a
        #     human player can see):
        #   player  : hp, block, energy                              (3)
        #   player powers : amount per POWER_VOCAB                   (P)
        #   enemies : MAX_ENEMIES slots x per-enemy block            (ME * E_FEATS)
        #             where each = alive, hp_frac, block, incoming_dmg,
        #                          is_attack, is_debuff, powers(P)
        #   hand    : MAX_HAND slots x (card one-hot V + cost + playable)
        #   piles   : draw/discard/exhaust counts over V             (3V)
        self.E_FEATS = 6 + P
        obs_dim = 3 + P + MAX_ENEMIES * self.E_FEATS + MAX_HAND * (V + 2) + 3 * V
        self.observation_space = spaces.Box(
            low=-1.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )

    # ── Gym API ─────────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Reuse one long-lived engine; reset_to_combat() does a fast in-process
        # combat reset (~1-2ms) after the first (one-time ~0.8s) setup.
        if self.engine is None:
            self.engine = Engine()
        # Pick a random encounter from the pool (self.np_random is seeded by Gym),
        # unless one is forced via options={"encounter": ...} (used by eval).
        if options and options.get("encounter"):
            encounter = options["encounter"]
        else:
            encounter = ENCOUNTER_POOL[self.np_random.integers(len(ENCOUNTER_POOL))]
        self.state = self.engine.reset_to_combat(encounter)
        self._prev_hp = float(self.state.get("player", {}).get("hp", MAX_HP))
        return self._encode(self.state), {}

    def step(self, action: int):
        action = int(action)
        if action == END_TURN_ACTION:
            self.state = self.engine.action("end_turn")
        else:
            card_slot, target_slot = divmod(action, MAX_ENEMIES)
            self.state = self._play(card_slot, target_slot)

        reward, terminated, outcome = self._reward_and_done(self.state)
        obs = self._encode(self.state)
        # `outcome` on the terminal step is "won" (enemy killed) or "died" (game over),
        # so callers can measure true survival separately from the reward score.
        info = {"outcome": outcome} if terminated else {}
        # `truncated` would be for hitting a step limit; we don't use one yet.
        return obs, reward, terminated, False, info

    def action_masks(self) -> np.ndarray:
        """Boolean array (len N_ACTIONS): which actions are legal this turn.

        For a playable card targeting AnyEnemy, every living enemy is a legal target.
        For a card that doesn't pick an enemy (Self/All/None), only target slot 0 is
        offered (the target is ignored), so the agent isn't shown redundant choices.
        MaskablePPO calls this so the policy never picks an illegal move."""
        mask = np.zeros(N_ACTIONS, dtype=bool)
        hand = self.state.get("hand") or []
        n_enemies = min(len(self.state.get("enemies") or []), MAX_ENEMIES)
        for i in range(min(len(hand), MAX_HAND)):
            card = hand[i]
            if not card.get("can_play"):
                continue
            base = i * MAX_ENEMIES
            if card.get("target_type") == "AnyEnemy":
                for j in range(n_enemies):
                    mask[base + j] = True
            else:
                mask[base + 0] = True   # canonical no-target action
        mask[END_TURN_ACTION] = True    # ending the turn is always legal
        return mask

    def close(self):
        if self.engine is not None:
            self.engine.close()
            self.engine = None

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _play(self, card_slot: int, target_slot: int) -> dict:
        hand = self.state.get("hand") or []
        if card_slot >= len(hand):
            return self.state  # illegal (mask should prevent this); no-op
        card = hand[card_slot]
        # AnyEnemy cards target the chosen living enemy; others ignore the target.
        if card.get("target_type") == "AnyEnemy":
            return self.engine.action("play_card", card_index=card_slot, target_index=target_slot)
        return self.engine.action("play_card", card_index=card_slot)

    def _reward_and_done(self, state: dict) -> tuple[float, bool, str | None]:
        decision = state.get("decision")
        # Player died → engine reports game_over (not a victory).
        if decision == "game_over" or state.get("type") == "error":
            hp = float(state.get("player", {}).get("hp", 0.0))
            reward = -0.5 * max(0.0, self._prev_hp - hp)
            return reward, True, "died"
        # Left combat (enemy dead) → we won this fight. Any decision that isn't
        # combat_play here means the combat resolved in our favor.
        if decision != "combat_play":
            return 10.0, True, "won"
        # Still fighting: penalize HP lost since the previous step.
        hp = float(state.get("player", {}).get("hp", self._prev_hp))
        reward = -0.5 * max(0.0, self._prev_hp - hp)
        self._prev_hp = hp
        return reward, False, None

    def _enemy_block(self, enemy: dict) -> np.ndarray:
        """Per-enemy feature block: alive, hp_frac, block, intent, powers."""
        e_max = max(1.0, float(enemy.get("max_hp", 1) or 1))
        is_attack, is_debuff = _intent_flags(enemy)
        scalars = np.array([
            1.0,                                              # alive flag
            float(enemy.get("hp", 0)) / e_max,
            float(enemy.get("block", 0)) / NORM_BLOCK,
            _incoming_damage(enemy) / MAX_HP,
            is_attack,
            is_debuff,
        ], dtype=np.float32)
        return np.concatenate([scalars, _power_vec(enemy.get("powers"))])

    def _encode(self, state: dict) -> np.ndarray:
        p = state.get("player", {})
        enemies = state.get("enemies") or []

        parts: list[np.ndarray] = []
        # player scalars
        parts.append(np.array([
            float(p.get("hp", 0)) / MAX_HP,
            float(p.get("block", 0)) / NORM_BLOCK,
            float(state.get("energy", 0)) / NORM_ENERGY,
        ], dtype=np.float32))
        # player status effects (e.g. Shrink)
        parts.append(_power_vec(state.get("player_powers")))
        # enemies: one fixed-size block per slot; empty slots are zeros (alive=0).
        for j in range(MAX_ENEMIES):
            if j < len(enemies):
                parts.append(self._enemy_block(enemies[j]))
            else:
                parts.append(np.zeros(self.E_FEATS, dtype=np.float32))
        # hand: MAX_HAND slots
        hand = state.get("hand") or []
        for i in range(MAX_HAND):
            if i < len(hand):
                c = hand[i]
                parts.append(_onehot_card(c))
                parts.append(np.array([
                    float(c.get("cost", 0)) / NORM_ENERGY,
                    1.0 if c.get("can_play") else 0.0,
                ], dtype=np.float32))
            else:
                parts.append(np.zeros(V + 2, dtype=np.float32))  # empty slot
        # piles
        parts.append(_pile_counts(state.get("draw_pile")))
        parts.append(_pile_counts(state.get("discard_pile")))
        parts.append(_pile_counts(state.get("exhaust_pile")))

        return np.concatenate(parts).astype(np.float32)


# Manual test: play one episode with a random *legal* policy and report the reward.
# `python rl/sts_env.py`
if __name__ == "__main__":
    env = StsCombatEnv()
    obs, _ = env.reset()
    print("observation vector length:", obs.shape[0])
    total, steps = 0.0, 0
    done = False
    rng = np.random.default_rng(0)
    while not done:
        mask = env.action_masks()
        legal = np.flatnonzero(mask)
        action = int(rng.choice(legal))     # pick a random *legal* action
        obs, reward, terminated, truncated, _ = env.step(action)
        total += reward
        steps += 1
        done = terminated or truncated
    print(f"episode finished in {steps} steps, total reward = {total:.1f}")
    env.close()
