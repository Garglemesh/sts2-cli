# sts2-rl — reinforcement learning on the headless engine

Train a neural network to play Slay the Spire 2 by driving the real game engine
headless. **Milestone 1: learn to win one fixed combat** (Ironclad vs. a single
weak enemy), then generalize from there.

## Layout

| File | Role |
|---|---|
| `engine.py` | Low-level driver: launches the C# engine, sends/reads JSON, resets to a fixed combat. No ML here. |
| `sts_env.py` | *(coming next)* Wraps `engine.py` as a Gymnasium environment: observation encoding, action space + masking, reward. |
| `train.py` | *(later)* Runs MaskablePPO to train the policy. |
| `requirements.txt` | Pinned Python deps (torch, gymnasium, stable-baselines3, sb3-contrib). |

## Reward (milestone 1)

- **+10** for winning the combat.
- **−0.5** for each point of HP lost during the combat.

So dying after losing all 80 HP ≈ −40; winning untouched = +10; winning at half
HP ≈ +10 − 0.5·40 = −10. The agent is pushed to win *and* preserve HP.

## What the agent observes

The engine exposes everything a human player can see, including the **contents of
the draw / discard / exhaust piles** (the draw pile's *order* is hidden, just like
in the real game). That lets the policy reason about what's left to draw.

## Running locally (host venv)

```bash
# one-time: deps already installed in rl/.venv
rl/.venv/bin/python rl/engine.py          # smoke-test the fixed-combat reset
```

## Running in Docker (portable to another ML box)

Prerequisites on the host: Docker + **nvidia-container-toolkit** (for `--gpus`).

```bash
# from the repo root
docker build -t sts2-rl .
docker run --rm -it --gpus all sts2-rl     # interactive shell inside the container
# inside: `python rl/engine.py`, later `python rl/train.py`
```

Or with compose: `docker compose run --rm trainer`.

Move the built image to another box without rebuilding:

```bash
docker save sts2-rl | gzip > sts2-rl.tar.gz
# scp to the other box, then:
gunzip -c sts2-rl.tar.gz | docker load
```

> The image bundles proprietary game DLLs from `lib/` — keep it private (your own
> machines only), don't push to a public registry.
