#!/usr/bin/env bash
set -eo pipefail

cd "$(dirname "$0")/.."
source /home/sra/medhansh/supermimic-official/.venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD/.upstream/supermimic/src:$PWD/.upstream/mjlab_research/src"

model_replay=${1:-recent}
case "$model_replay" in
  recent|history) ;;
  *) echo "Expected recent or history" >&2; exit 2 ;;
esac
yaw_reward=${2:-clipped}
cube_mm=${3:-42}
iterations=${SUPERDEX_ITERATIONS:-500}
actor_horizon=${SUPERDEX_ACTOR_HORIZON:-4}
seed=${SUPERDEX_SEED:-42}
demo_args=()
eval_steps=150
suffix=""
if [[ "$yaw_reward" == turn ]]; then
  eval_steps=400
  suffix="-turn"
elif [[ "$yaw_reward" == quarter ]]; then
  eval_steps=200
  suffix="-quarter"
fi
if [[ -n "${4:-}" ]]; then
  demo_args=(--demo-bank "$4")
  suffix="$suffix-demo"
fi
root=/mnt/data1/sra/medhansh/superdex
export TMPDIR="$root/tmp" XDG_CACHE_HOME="$root/cache"
export WARP_CACHE_PATH="$root/cache/warp" CUDA_CACHE_PATH="$root/cache/cuda"
export TORCHINDUCTOR_CACHE_DIR="$root/cache/torchinductor" TRITON_CACHE_DIR="$root/cache/triton"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$root/runs" "$TMPDIR" "$XDG_CACHE_HOME"
run=$(mktemp -d "$root/runs/joint-${cube_mm}mm-i${iterations}-$model_replay$suffix-$(date +%Y%m%d-%H%M%S)-XXXXXX")
echo "Run directory: $run"
git rev-parse HEAD > "$run/commit.txt"
python -u -m superdex.train --device cuda:0 --cube-mm "$cube_mm" --envs 32 --seed "$seed" \
  --iterations "$iterations" --actor-horizon "$actor_horizon" --world-model-loss original --yaw-reward "$yaw_reward" \
  --model-replay "$model_replay" --policy-noise-std 0.05 --report-every 25 --save-reports --eval-steps "$eval_steps" \
  "${demo_args[@]}" --output "$run/joint.pt" 2>&1 | tee "$run/train.log"
python scripts/summarize-run.py "$run"
