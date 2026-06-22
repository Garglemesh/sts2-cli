"""
train.py — train a policy to win the fixed combat with MaskablePPO.

Big picture: PPO ("Proximal Policy Optimization") is a popular RL algorithm. It
repeatedly (1) plays a bunch of combats with the current policy, (2) looks at which
actions led to more reward, and (3) nudges the neural network toward those actions.
"Maskable" = it respects our action_masks(), so it never even considers illegal moves.

stable-baselines3 / sb3-contrib implement all of that. Our job is just to hand it
the environment and sensible settings, then watch the average reward climb.

Run:
    python rl/train.py                 # train with defaults
    python rl/train.py --timesteps 200000 --envs 8

Watch the line `ep_rew_mean` in the output — that's the average reward per combat.
It should rise from around random (~ -6) toward the +10 win ceiling.
"""
from __future__ import annotations

import argparse
import os

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from sts_env import StsCombatEnv

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")


def make_env():
    """Factory the vectorized wrapper calls to build one env per worker process."""
    def _init():
        return StsCombatEnv()
    return _init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=50_000,
                    help="total environment steps to train for")
    ap.add_argument("--envs", type=int, default=8,
                    help="parallel engine processes gathering experience")
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    ap.add_argument("--out", default="combat_ppo", help="checkpoint name")
    args = ap.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Several engines play in parallel so we collect experience faster. Each runs in
    # its own OS process (SubprocVecEnv) and owns one dotnet engine.
    venv = SubprocVecEnv([make_env() for _ in range(args.envs)])
    # VecMonitor records each finished episode's reward/length so sb3 can log
    # `ep_rew_mean` (the number we watch to see learning happen).
    venv = VecMonitor(venv)

    model = MaskablePPO(
        "MlpPolicy",          # a small fully-connected network for our 78-number obs
        venv,
        verbose=1,
        device=args.device,
        n_steps=256,          # steps each env collects before a learning update
        batch_size=256,
        gamma=0.99,           # how much future reward matters vs immediate
        tensorboard_log=os.path.join(CHECKPOINT_DIR, "tb"),
    )

    print(f"Training {args.timesteps} steps across {args.envs} envs on {args.device}…")
    print("Watch `ep_rew_mean` — it should climb from ~ -6 toward +10.\n")
    model.learn(total_timesteps=args.timesteps, progress_bar=False)

    out_path = os.path.join(CHECKPOINT_DIR, args.out)
    model.save(out_path)
    print(f"\nSaved trained policy to {out_path}.zip")
    venv.close()


if __name__ == "__main__":
    main()
