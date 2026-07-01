#!/usr/bin/env bash
# Launch QAD (Quantization-Aware Distillation) for Qwen3-8B on the reasoning-mix blend.
#
# Usage:
#   ./run_qad.sh                         # all visible GPUs, defaults below
#   CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_qad.sh
#   WANDB_PROJECT=my-proj WANDB_NAME=my-run ./run_qad.sh
#   ./run_qad.sh --learning_rate 5e-6    # any extra flag is forwarded to train.py
#   nohup ./run_qad.sh > qad_train.log 2>&1 &   # detached with a log file

# --- editable defaults (override via env when calling) ---
: "${WANDB_PROJECT:=qwen3-8b-qad}"                 # wandb project (env-only; the yaml `project` field is Trackio-only)
: "${WANDB_NAME:=qwen3-8b-qad-nvfp4-reasonmix}"    # wandb run name
export WANDB_PROJECT WANDB_NAME
# export WANDB_ENTITY=your-team                    # uncomment to set the wandb team/org

ACCEL_CFG=configs/accelerate/fsdp2.yaml            # FSDP2; swap for ddp.yaml / deepspeed.yaml if desired
TRAIN_CFG=configs/train/qad_nvfp4_reasonmix.yaml
STUDENT=/home/zhizhousha/workspace/low-precision-project/model-and-data/models/Qwen3-8B-modelopt-qad-quantized-student
TEACHER=/home/zhizhousha/workspace/low-precision-project/model-and-data/models/Qwen3-8B

# --- go to this script's own dir (examples/llm_qat) so relative config paths resolve ---
cd "$(dirname "$(readlink -f "$0")")"

# --- activate the venv (sets Triton/CUDA env via the venv's .pth, and PATH) ---
source ../../.venv/bin/activate
set -euo pipefail

# --- preflight: the quantized student and the teacher must exist ---
if [ ! -f "$STUDENT/modelopt_state.pth" ]; then
  echo "ERROR: quantized student not found at:"
  echo "       $STUDENT"
  echo "Run quantize.py first (step 1) to produce it."
  exit 1
fi
if [ ! -f "$TEACHER/config.json" ]; then
  echo "ERROR: teacher model not found at: $TEACHER"
  exit 1
fi

echo "=============================================================="
echo " QAD launch"
echo "   accelerate cfg : $ACCEL_CFG"
echo "   train cfg      : $TRAIN_CFG"
echo "   student        : $STUDENT"
echo "   teacher        : $TEACHER"
echo "   WANDB_PROJECT  : $WANDB_PROJECT"
echo "   WANDB_NAME     : $WANDB_NAME"
echo "   CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES:-<all visible GPUs>}"
echo "=============================================================="

# --- launch (extra CLI args are forwarded to train.py and override the yaml) ---
accelerate launch --config-file "$ACCEL_CFG" train.py \
  --config "$TRAIN_CFG" "$@"
