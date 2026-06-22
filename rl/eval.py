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
from engine import ENCOUNTER_POOL

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")


def run_episodes(model, env, n, encounter=None):
    """Play n episodes (optionally forcing one encounter); return per-episode stats."""
    survived, rewards, hp_left = 0, [], []
    for _ in range(n):
        opts = {"encounter": encounter} if encounter else None
        obs, _ = env.reset(options=opts)
        total, done, outcome = 0.0, False, None
        while not done:
            action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
            obs, r, term, trunc, info = env.step(int(action))
            total += r
            done = term or trunc
            if done:
                outcome = info.get("outcome")
        rewards.append(total)
        if outcome == "won":
            survived += 1
            hp_left.append(MAX_HP - (10.0 - total) / 0.5)
    return survived, rewards, hp_left


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="combat_ppo")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--by-encounter", action="store_true",
                    help="break results down per encounter in the pool")
    args = ap.parse_args()

    model = MaskablePPO.load(os.path.join(CHECKPOINT_DIR, args.model))
    env = StsCombatEnv()

    if args.by_encounter:
        print(f"{'encounter':24} {'survive':>9}  {'HP_on_win':>9}  {'avg_reward':>10}")
        for enc in ENCOUNTER_POOL:
            surv, rewards, hp = run_episodes(model, env, args.episodes, encounter=enc)
            hp_str = f"{np.mean(hp):.0f}/{int(MAX_HP)}" if hp else "-"
            print(f"{enc:24} {surv}/{args.episodes:>3}={100*surv//args.episodes:>3}%  "
                  f"{hp_str:>9}  {np.mean(rewards):>+10.1f}")
    else:
        n = args.episodes
        surv, rewards, hp = run_episodes(model, env, n)
        print(f"episodes:          {n}")
        print(f"survived (won):    {surv}/{n} = {100*surv/n:.0f}%")
        print(f"avg reward:        {np.mean(rewards):+.2f}  (std {np.std(rewards):.2f})")
        if hp:
            print(f"avg HP on win:     {np.mean(hp):.1f}/{int(MAX_HP)}")
        print(f"deaths:            {n - surv}/{n}")
    env.close()


if __name__ == "__main__":
    main()
