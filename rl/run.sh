#!/usr/bin/env bash
# Convenience launcher so you don't type the venv path every time.
#   ./rl/run.sh watch --encounter EXOSKELETONS_WEAK
#   ./rl/run.sh eval  --model combat_ppo_phase2 --by-encounter
#   ./rl/run.sh train --timesteps 300000 --out my_run
#   ./rl/run.sh tensorboard           # launches the dashboard on :6006
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"   # repo root
py="$here/rl/.venv/bin/python"

cmd="${1:?usage: ./rl/run.sh <watch|eval|train|tensorboard> [args...]}"
shift || true

if [ "$cmd" = "tensorboard" ]; then
    exec "$here/rl/.venv/bin/tensorboard" --logdir "$here/checkpoints/tb" "$@"
fi
exec "$py" "$here/rl/$cmd.py" "$@"
