"""
watch.py — watch a trained policy play combats, turn by turn, in plain English.

    python rl/watch.py                                  # 2 random encounters
    python rl/watch.py --model combat_ppo_phase2 --episodes 3
    python rl/watch.py --encounter EXOSKELETONS_WEAK    # force one fight
"""
from __future__ import annotations

import argparse
import os

from sb3_contrib import MaskablePPO

from sts_env import StsCombatEnv, MAX_ENEMIES, END_TURN_ACTION

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")


def describe_state(s: dict) -> str:
    p = s.get("player", {})
    line = f"  You: {p.get('hp')}/{p.get('max_hp')} HP, {p.get('block')} block, {s.get('energy')} energy"
    pw = s.get("player_powers") or []
    if pw:
        line += "  [" + ", ".join(f"{w['name']} {w['amount']}" for w in pw) + "]"
    parts = [line]
    for i, e in enumerate(s.get("enemies") or []):
        intent = ""
        for it in e.get("intents") or []:
            if it.get("damage") is not None:
                intent = f" -> attacks for {it.get('total_damage') or it.get('damage')}"
            elif it.get("type"):
                intent = f" -> {it['type']}"
        epw = e.get("powers") or []
        ptxt = "  [" + ", ".join(f"{w['name']} {w['amount']}" for w in epw) + "]" if epw else ""
        parts.append(f"  Enemy {i}: {e.get('name')} {e.get('hp')}/{e.get('max_hp')} HP{intent}{ptxt}")
    return "\n".join(parts)


def describe_action(action: int, s: dict) -> str:
    if action == END_TURN_ACTION:
        return "END TURN"
    card_slot, target = divmod(action, MAX_ENEMIES)
    hand = s.get("hand") or []
    name = hand[card_slot]["name"] if card_slot < len(hand) else f"slot{card_slot}"
    if card_slot < len(hand) and hand[card_slot].get("target_type") == "AnyEnemy":
        enemies = s.get("enemies") or []
        tgt = enemies[target]["name"] if target < len(enemies) else f"enemy{target}"
        return f"play {name} -> {tgt}"
    return f"play {name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="combat_ppo_phase2")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--encounter", default=None)
    args = ap.parse_args()

    model = MaskablePPO.load(os.path.join(CHECKPOINT_DIR, args.model))
    env = StsCombatEnv()

    for ep in range(args.episodes):
        opts = {"encounter": args.encounter} if args.encounter else None
        obs, _ = env.reset(options=opts)
        print(f"\n{'='*64}\nEPISODE {ep+1}  —  {env.engine._encounter}\n{'='*64}")
        total, done, turn = 0.0, False, 0
        print(describe_state(env.state))
        while not done:
            action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
            action = int(action)
            print(f"    > {describe_action(action, env.state)}")
            obs, r, term, trunc, info = env.step(action)
            total += r
            done = term or trunc
            if not done and action == END_TURN_ACTION:
                turn += 1
                print(f"  -- turn {turn} --\n{describe_state(env.state)}")
            if done:
                print(f"\n  RESULT: {info.get('outcome','?').upper()}  |  reward {total:+.1f}  |  "
                      f"HP {env.state.get('player',{}).get('hp','?')}")
    env.close()


if __name__ == "__main__":
    main()
