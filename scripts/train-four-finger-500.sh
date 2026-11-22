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
eval_steps=150
suffix=""
if [[ "$yaw_reward" == turn ]]; then
  eval_steps=400
  suffix="-turn"
fi
root=/mnt/data1/sra/medhansh/superdex
export TMPDIR="$root/tmp" XDG_CACHE_HOME="$root/cache"
export WARP_CACHE_PATH="$root/cache/warp" CUDA_CACHE_PATH="$root/cache/cuda"
export TORCHINDUCTOR_CACHE_DIR="$root/cache/torchinductor" TRITON_CACHE_DIR="$root/cache/triton"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$root/runs" "$TMPDIR" "$XDG_CACHE_HOME"
run=$(mktemp -d "$root/runs/joint-42mm-i500-$model_replay$suffix-$(date +%Y%m%d-%H%M%S)-XXXXXX")
echo "Run directory: $run"
git rev-parse HEAD > "$run/commit.txt"
python -u -m superdex.train --device cuda:0 --cube-mm 42 --envs 32 --seed 42 \
  --iterations 500 --actor-horizon 4 --world-model-loss original --yaw-reward "$yaw_reward" \
  --model-replay "$model_replay" --policy-noise-std 0.05 --report-every 25 --save-reports --eval-steps "$eval_steps" \
  --output "$run/joint.pt" 2>&1 | tee "$run/train.log"
python scripts/summarize-run.py "$run"
