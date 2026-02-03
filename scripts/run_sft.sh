#!/bin/bash

# ================== Environment Variables ==================
export PYTHONPATH=src:$PYTHONPATH
export FORCE_QWENVL_VIDEO_READER=decord
DEVICES_LIST="0,1,2,3,4,5,6,7"

# ================== Model / Data / Output Paths ==================
ROOT_DIR=$(dirname "$(dirname "$0")")
MODEL_PATH=/cache/ckpt/Qwen/Qwen2.5-VL-3B-Instruct
DATA_ROOT=/path/to/data_root
DATA_JSON=./data/demo.json  # Replace with your dataset json if needed

# ======================= Choose config ========================
PPE_CONFIG="${ROOT_DIR}/scripts/configs/ppe.json"
# PPE_CONFIG="${ROOT_DIR}/scripts/configs/ppe_cascade.json"
# PPE_CONFIG="${ROOT_DIR}/scripts/configs/chatunivi.json"
# PPE_CONFIG="${ROOT_DIR}/scripts/configs/dense.json"


if [[ "$1" == "debug" ]]; then
    DEBUG_MODE=true
    OUTPUT_DIR=/cache/ckpt/qwen2.5_vl_3b/DEBUG
else
    DEBUG_MODE=false
    OUTPUT_DIR=/cache/ckpt/qwen2.5_vl_3b/demo_ppe/$(date +%Y%m%d_%H%M)
fi
mkdir -p "$OUTPUT_DIR"

# ================== Save Script Config / Log ==================
CONFIG_DIR="${OUTPUT_DIR}/training_config"
mkdir -p "$CONFIG_DIR"
cp "$0" "${CONFIG_DIR}/script.sh"

LOGFILE="${OUTPUT_DIR}/train.log"

# ================== Detect Device Type ==================
if command -v nvidia-smi &>/dev/null; then
  export DEVICE_TYPE="GPU"
  DEVICE_ENV="CUDA_VISIBLE_DEVICES"
  DEFAULT_DEVICES=$DEVICES_LIST
  DISABLE_FLASH_ATTN2=False # default for GPU
  USE_LIGER=True            # default for GPU
  TF32=True                 # default for GPU
  echo "[INFO] Detected NVIDIA GPU environment."
elif command -v npu-smi &>/dev/null || [[ -d "/usr/local/Ascend" ]]; then
  export DEVICE_TYPE="NPU"
  DEVICE_ENV="ASCEND_RT_VISIBLE_DEVICES"
  DEFAULT_DEVICES=$DEVICES_LIST
  DISABLE_FLASH_ATTN2=True # default for NPU
  USE_LIGER=False          # default for NPU
  TF32=False               # default for NPU
  echo "[INFO] Detected Ascend NPU environment."
else
  echo "[ERROR] No supported device detected (GPU or NPU)."
  exit 1
fi

# ================== Deepspeed Configuration ==================
DEEPSPEED=scripts/zero2_offload.json
cp "$DEEPSPEED" "${CONFIG_DIR}/config_$(basename "$DEEPSPEED")"

# ================== Training Settings ==================
GLOBAL_BATCH_SIZE=32
BATCH_PER_DEVICE=1

if $DEBUG_MODE; then
    export ${DEVICE_ENV}=0
    NUM_DEVICES=1
    echo "[DEBUG] Running sft on single $DEVICE_TYPE (device 0)"
else
    export ${DEVICE_ENV}=${DEFAULT_DEVICES}
    NUM_DEVICES=$(echo ${DEFAULT_DEVICES} | awk -F',' '{print NF}')
    echo "[INFO] Running sft on ${NUM_DEVICES} ${DEVICE_TYPE}s (${DEFAULT_DEVICES})"
fi

# Calculate gradient accumulation steps
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))
if ((GRAD_ACCUM_STEPS <= 0)); then
    echo "[ERROR] GRAD_ACCUM_STEPS <= 0, check GLOBAL_BATCH_SIZE / BATCH_PER_DEVICE / NUM_DEVICES"
    exit 1
fi

# ================== Common Training Arguments ==================
COMMON_ARGS="\
    --use_liger $USE_LIGER \
    --model_id $MODEL_PATH \
    --data_path $DATA_JSON \
    --image_folder $DATA_ROOT \
    --remove_unused_columns False \
    --freeze_vision_tower False \
    --freeze_llm False \
    --tune_merger True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 $DISABLE_FLASH_ATTN2 \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((512 * 28 * 28)) \
    --image_max_pixels $((1280 * 28 * 28)) \
    --video_min_pixels $((128 * 28 * 28)) \
    --video_max_pixels $((768 * 28 * 28)) \
    --fps 2.0 \
    --nframes 64 \
    --learning_rate 1e-6 \
    --merger_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 $TF32 \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 1 \
    --dataloader_num_workers 8 \
    --ppe_config=${PPE_CONFIG} \
"

# ================== Start Training ==================
if $DEBUG_MODE; then
    echo "[DEBUG] Running training in single GPU mode"
    python src/training/train.py $COMMON_ARGS 2>&1 | tee "$LOGFILE"
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
else
    echo "[TRAIN] Running training with DeepSpeed"
    deepspeed src/training/train.py --deepspeed "$DEEPSPEED" $COMMON_ARGS 2>&1 | tee "$LOGFILE"
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
fi

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "[INFER] Training completed"
else
    echo "[ERROR] Training failed (exit code = $TRAIN_EXIT_CODE)"
    exit 1
fi
