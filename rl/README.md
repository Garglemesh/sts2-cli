# sts2-rl — reinforcement learning on the headless engine

Train a neural network to play Slay the Spire 2 by driving the real game engine
headless, using MaskablePPO (sb3-contrib).

## Layout

| File | Role |
|---|---|
| `engine.py` | Low-level driver: launches the compiled C# engine, JSON IO, fast in-process combat reset. No ML here. |
| `sts_env.py` | Gymnasium environment: observation encoding, action space + masking, reward. |
| `train.py` | Trains a MaskablePPO policy over parallel engines. `--timesteps N --envs 8 --out NAME` |
| `eval.py` | Honest evaluation of a checkpoint. `--model NAME --episodes N [--by-encounter]` |
| `requirements.txt` | Pinned Python deps (torch, gymnasium, stable-baselines3, sb3-contrib, tensorboard). |

## Progress & how to resume

Current state (branch `rl-trainer`):

- **Phase 1 — fixed combat (Ironclad vs 1 weak beetle):** survives 100% (0 deaths);
  enriched observation (intent damage + status effects) → ~89% of wins also keep >20 HP.
  Checkpoint: `checkpoints/combat_ppo`.
- **Phase 2 — multiple enemies + target selection, random encounters (1–3 enemies):**
  action space `(card × target) + end turn`; obs 141-dim with up to 5 enemy slots.
  **Survives 100% of all 6 encounters, 0 deaths.** Costly only on tanky 90+HP fights
  (BOWLBUGS/CULTISTS) where the bare starter deck can't kill fast — a *deck* limit.
  Checkpoint: `checkpoints/combat_ppo_phase2`.
- **Next — Phase 3: deck-building / meta-game.** Keep card rewards between fights so the
  agent gets stronger over a run (fixes the HP-efficiency gap; the heart of real StS).

Key infra: episode reset is **in-process** (~1.5ms) — `start_run` once per engine, then
restore HP+deck and `enter_room` a fresh fight. This gave ~700 fps / a 21× training
speedup (50k steps in ~45s). See `engine.py`.

Quick commands:

```bash
export STS2_GAME_DIR="$PWD/lib"                      # engine finds the game DLLs
rl/.venv/bin/python rl/engine.py                     # smoke-test a combat reset
rl/.venv/bin/python rl/train.py --timesteps 300000 --envs 8 --out my_run
rl/.venv/bin/python rl/eval.py --model my_run --episodes 30 --by-encounter
```

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
