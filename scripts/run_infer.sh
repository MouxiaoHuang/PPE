#!/bin/bash
set -x  # Print commands for easier debugging


# ================== Path Setup ==================
ROOT_DIR=$(dirname "$(dirname "$0")")
INFER_SCRIPT="src.evaluate.inference"
DEVICES_LIST="0,1,2,3,4,5,6,7"

MODEL_PATH=${MODEL_PATH:-/cache/ckpt/Qwen/Qwen2.5-VL-3B-Instruct}
BENCHMARKS="videomme videomme_sub mvbench seedbench_video nextqa_mc nextqa_oe"
# BENCHMARKS="mmbench_en mmbench_cn sqa textvqa docvqa ocrbench chartqa"

# ======================= Choose config ========================
PPE_CONFIG="${ROOT_DIR}/scripts/configs/ppe.json"
# PPE_CONFIG="${ROOT_DIR}/scripts/configs/ppe_cascade.json"
# PPE_CONFIG="${ROOT_DIR}/scripts/configs/chatunivi.json"
# PPE_CONFIG="${ROOT_DIR}/scripts/configs/dense.json"


if [[ ! -d "$MODEL_PATH" ]]; then
  echo "[ERROR] MODEL_PATH does not exist: $MODEL_PATH"
  exit 1
fi

# ================== Inference Modes ==================
DEBUG_MODE=false

if [[ "$1" == "debug" ]]; then
  DEBUG_MODE=true
fi

# ================== Detect Device Type ==================
if command -v nvidia-smi &>/dev/null; then
  export DEVICE_TYPE="GPU"
  DEVICE_ENV="CUDA_VISIBLE_DEVICES"
  DEFAULT_DEVICES=$DEVICES_LIST
  echo "[INFO] Detected NVIDIA GPU environment."
elif command -v npu-smi &>/dev/null || [[ -d "/usr/local/Ascend" ]]; then
  export DEVICE_TYPE="NPU"
  DEVICE_ENV="ASCEND_RT_VISIBLE_DEVICES"
  DEFAULT_DEVICES=$DEVICES_LIST
  echo "[INFO] Detected Ascend NPU environment."
else
  echo "[ERROR] No supported device detected (GPU or NPU)."
  exit 1
fi

# ================== Inference Settings ==================
NFRAMES=64
VIDEO_MAX_PIXELS=$((768 * 28 * 28))

# ================== Device Setup ==================
if $DEBUG_MODE; then
  TIMESTAMP="DEBUG"
  export ${DEVICE_ENV}=0
  NUM_DEVICES=1
  echo "[DEBUG] Running inference on single $DEVICE_TYPE (device 0)"
else
  export ${DEVICE_ENV}=${DEFAULT_DEVICES}
  NUM_DEVICES=$(echo ${DEFAULT_DEVICES} | awk -F',' '{print NF}')
  TIMESTAMP=$(date +%Y%m%d_%H%M)
  echo "[INFO] Running inference on ${NUM_DEVICES} ${DEVICE_TYPE}s (${DEFAULT_DEVICES})"
fi

# ================== Launch Inference ==================
for task in $BENCHMARKS; do
  echo "[INFO] Running task: $task"

  for (( DEV_ID=0; DEV_ID<NUM_DEVICES; DEV_ID++ )); do
    echo "[INFO] Launching task $task on ${DEVICE_TYPE} $DEV_ID"

    CMD="${DEVICE_ENV}=${DEV_ID} python -m $INFER_SCRIPT \
      --script_path=\"$0\" \
      --model_path=${MODEL_PATH} \
      --num_gpus=${NUM_DEVICES} \
      --timestamp=${TIMESTAMP} \
      --nframes=${NFRAMES} \
      --ppe_config=${PPE_CONFIG} \
      --video_max_pixels=${VIDEO_MAX_PIXELS} \
      --task=${task} \
      --local_rank=${DEV_ID}"

    if $DEBUG_MODE; then
      eval "$CMD"
    else
      eval "$CMD &"
    fi
  done

  wait
done


# ================== Final Message ==================
if ! $DEBUG_MODE; then
  echo "[INFO] All inference tasks completed."
fi
