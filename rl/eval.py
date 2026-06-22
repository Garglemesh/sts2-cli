"""
eval.py — measure how good a trained policy actually is.

Training reward (ep_rew_mean) is a noisy running average over *exploring* play.
For an honest score we load the saved policy and play N combats with it acting
*greedily* (its single best move each turn, no exploration), then report win rate
and average HP remaining.

    python rl/eval.py                       # eval the default checkpoint
    python rl/eval.py --episodes 100
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from sb3_contrib import MaskablePPO

from sts_env import StsCombatEnv, MAX_HP

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="combat_ppo")
    ap.add_argument("--episodes", type=int, default=50)
    args = ap.parse_args()

    model = MaskablePPO.load(os.path.join(CHECKPOINT_DIR, args.model))
    env = StsCombatEnv()

    wins, rewards, hp_left = 0, [], []
    for _ in range(args.episodes):
        obs, _ = env.reset()
        total, done = 0.0, False
        while not done:
            mask = env.action_masks()
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, r, term, trunc, _ = env.step(int(action))
            total += r
            done = term or trunc
        rewards.append(total)
        won = total > 0          # a loss can't score positive under +10/-0.5
        wins += int(won)
        if won:
            # final player HP for won fights = MAX_HP minus total damage taken
            hp_left.append(MAX_HP - (10.0 - total) / 0.5)
    env.close()

    n = args.episodes
    print(f"episodes:        {n}")
    print(f"win rate:        {wins}/{n} = {100*wins/n:.0f}%")
    print(f"avg reward:      {np.mean(rewards):+.2f}  (std {np.std(rewards):.2f})")
    if hp_left:
        print(f"avg HP on win:   {np.mean(hp_left):.1f}/{int(MAX_HP)}")


if __name__ == "__main__":
    main()
