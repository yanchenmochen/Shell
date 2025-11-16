#!/bin/bash
set -eo pipefail
set -x

export GLOO_SOCKET_IFNAME=bond0
export PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True
export MUSA_VISIBLE_DEVICES=${MUSA_VISIBLE_DEVICES:-'0,1,2,3,4,5,6,7'}
export MUSA_KERNEL_TIMEOUT=3200000
export ACCELERATOR_BACKEND="musa"
export MCCL_PROTOS=2
export MCCL_CHECK_POINTERS=0
export OMP_NUM_THREADS=4
export MCCL_ALGOS=1
export MCCL_BUFFSIZE=20971520
export MUSA_BLOCK_SCHEDULE_MODE=1
export MCCL_IB_GID_INDEX=3
export MCCL_NET_SHARED_BUFFERS=0
export MCCL_IB_QPS_PER_CONNECTION=16
export MCCL_CROSS_NIC=0
export MCCL_IB_HCA=mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5
export MCCL_IB_TC=136
export MCCL_IB_TIMEOUT=20
export MCCL_IB_RETRY_CNT=7
export ENABLE_ZERO_BUBBLE=0 # if set 1, Enable zero_bubble
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CPU_OPTIMIZER_PRECISION_AWARE_RECONFIG=${CPU_OPTIMIZER_PRECISION_AWARE_RECONFIG:-0}
export USE_RECOMPUTE_VARIANCE=1
export ENABLE_D2H_IN_PERMUTATION=0
export NO_LOSS_REDUCE=0
export LOGLEVEL="INFO"
export MUSA_EXECUTION_TIMEOUT=20000000
export MUSA_BLOCK_DISTRIBUTION_GRANULARITY=0
#export MUSA_LOG=0x1

MEGATRON_PATH=/mnt/seed17/001688/wangluping/sft/Megatron-LM_kuae_1022
MEGATRON_MUSA_PATH=/mnt/seed17/001688/wangluping/sft/megatron-lm-musa-patch_kuae_1022

export PYTHONPATH=${MEGATRON_PATH}:${MEGATRON_MUSA_PATH}:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/openmpi/lib:/usr/local/musa/lib:${LD_LIBRARY_PATH}

if [ ! -d "${MEGATRON_PATH}/build" ]; then
  cd "${MEGATRON_PATH}"
  python setup.py build_ext --inplace
  cd -
fi

TP=${TP:-1}
PP=${PP:-1}
EP=${EP:-4}
DIST_BACKEND=${DIST_BACKEND:-nccl}
PR=${PR:-bf16}
FP8=${FP8:-false}
SFT=${SFT:-true}
GC=${GC:-1}
OVERLAP=${OVERLAP:-0}
TIMER_PRINT={TIMER_PRINT:-false}
TRAIN_ITERS=${TRAIN_ITERS:-3349}
CHECKPOINT_LOAD_PATH=${CHECKPOINT_LOAD_PATH:-"/mnt/moer-train/public/checkpoints/021-16B-8t-bf16-1108-compare-new"}

echo "Using MUSA backend"

ROUTER_DTYPE=${ROUTER_DTYPE:-fp32}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
SEQ_LEN=${SEQ_LEN:-4096}

#DATA_PATH=${DATA_PATH:-"/mnt/seed17/001688/wangluping/test-data/qwen_sft_negeos_packing_text_document"}
DATA_PATH=${DATA_PATH:-"/mnt/seed-program-nas/001688/datasets/SFT/tulu-3-sft-mixture/tulu_v3_mix_zjllm_tokenizer_negeos_packing_text_document"}
DATA_CACHE_PATH=/mnt/seed17/001688/wangluping/test-data/datacache
TOKENIZED_MODEL=/mnt/moer-train/public/models/zjllm-llama3-tokenizer
MODEL_NAME=${MODEL_NAME:-'021-16B-sft'}

if [[ -z ${SEEN_STEPS} ]]; then
  SEEN_STEPS=0
fi

#TRAIN_SAMPLES=${TRAIN_SAMPLES:-139040000}

TRAIN_SAMPLES=$(($TRAIN_ITERS * $GLOBAL_BATCH_SIZE))

WARMUP_STEPS=2000
WARMUP_SAMPLES=$((WARMUP_STEPS * 1600))
OUTPUT_DIR=${OUTPUT_DIR:-"/mnt/seed17/001688/wangluping/output/debug"}
unset MLFLOW_TRACKING_URI

###########################
###### change for multinode config
###########################
NUM_NODES=${WORLD_SIZE:-1}
CURRENT_TIME=$(date "+%Y-%m-%d_%H:%M")
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=9958
GPUS_PER_NODE=${TQ_GPU_NUM:-8}
RECOMPUTE_LAYERS=1
WORLD_SIZE=$((GPUS_PER_NODE * NUM_NODES))
DP_SIZE=$((WORLD_SIZE / (PP * TP)))
NODE_RANK=${RANK:-0}
SAVE_INTERVAL=${SAVE_INTERVAL:-1116}
EXIT_INTERVAL=${EXIT_INTERVAL:-4000}
###########################
EXPNAME="zj_tp${TP}_pp${PP}_dp${DP_SIZE}_ep${EP}_mbs${MICRO_BATCH_SIZE}_numbs${NUM_MICROBATCHES}_gbs${GLOBAL_BATCH_SIZE}_gpus${WORLD_SIZE}_router${ROUTER_DTYPE}"

MODEL_SIZE=${MODEL_SIZE:-"A2.4B"}

if [ $MODEL_SIZE = A2.4B ]; then
  HIDDEN_SIZE=2048
  NUM_ATTN_HEADS=16
  NUM_LAYERS=28
  INTERMEDIATE_SIZE=10944
  MOE_INTERMEDIATE_SIZE=1408
  MAX_POSITION_EMBEDDINGS=${SEQ_LEN}
  EXTRA_VOCAB_SIZE=2400
  KV_LORA_RANK=512
  Q_LORA_RANK=1536
  QK_NOPE_HEAD_DIM=128
  QK_ROPE_HEAD_DIM=64
  V_HEAD_DIM=128
  ROPE_THETA=10000
  SCALE_FACTOR=1
  NUM_EXPERTS=64
  ROUTER_TOPK=6
  NUM_SHARED_EXPERTS=2
  MOE_LAYER_FREQ=1
  RMS_NORM_EPS=1e-6
else
  # 021-32B最新参数
  HIDDEN_SIZE=2048
  NUM_ATTN_HEADS=32
  # 021-32B 为40层
  NUM_LAYERS=40
  INTERMEDIATE_SIZE=12288
  MOE_INTERMEDIATE_SIZE=1536
  MAX_POSITION_EMBEDDINGS=4096
  Q_LORA_RANK=1536
  KV_LORA_RANK=512
  QK_NOPE_HEAD_DIM=128
  QK_ROPE_HEAD_DIM=64
  V_HEAD_DIM=128
  ROPE_THETA=10000
  SCALE_FACTOR=1
  NUM_EXPERTS=80
  ROUTER_TOPK=7
  NUM_SHARED_EXPERTS=1
  MOE_LAYER_FREQ=1
  MOE_FIRST_K_DENSE_REPLACE=1
  RMS_NORM_EPS=1e-6
fi

if [ "$PP" -eq 1 ]; then
  LAST_STAGE_ARG=""
else
  # 计算：每个 stage 层数（商），余下层加到最后一个 stage
  LAYERS_PER_STAGE=$((NUM_LAYERS / $PP))
  REMAINING_LAYERS=$((NUM_LAYERS % $PP))
  LAST_STAGE=$((LAYERS_PER_STAGE + REMAINING_LAYERS))

  # 设置最后一个 stage 层数参数
  LAST_STAGE_ARG="--decoder-last-pipeline-num-layers ${LAST_STAGE}"
fi

# 获取当前机器IP
NODE_ADDR=$(ip a | awk '/inet / && !/127.0.0.1/ {print $2}' | cut -d/ -f1 | head -n 1)

echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "NODE_ADDR: ${NODE_ADDR}"
echo "NODE_RANK: ${NODE_RANK}"

mkdir -p ${OUTPUT_DIR}/logs/${CURRENT_TIME}
LOG_FILE="${OUTPUT_DIR}/logs/${CURRENT_TIME}/${EXPNAME}.${NODE_RANK}.${NODE_ADDR}.log"
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"${OUTPUT_DIR}/checkpoint/${EXPNAME}"}
mkdir -p ${CHECKPOINT_PATH}
TENSORBOARD_PATH="${OUTPUT_DIR}/tf_logs/${CURRENT_TIME}_${MODEL_NAME}"

###########################
###### training args
###########################
DISTRIBUTED_ARGS=(
  --nproc_per_node $GPUS_PER_NODE
  --nnodes $NUM_NODES
  --node_rank $NODE_RANK
  --master_addr $MASTER_ADDR
  --master_port $MASTER_PORT
)

MODEL_ARGS=(
  --num-layers $NUM_LAYERS # 60
  --hidden-size ${HIDDEN_SIZE}
  --num-attention-heads $NUM_ATTN_HEADS
  --seq-length ${SEQ_LEN}
  --max-position-embeddings $MAX_POSITION_EMBEDDINGS
  --norm-epsilon $RMS_NORM_EPS
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --disable-bias-linear
  --ffn-hidden-size $INTERMEDIATE_SIZE
  --position-embedding-type rope
  --rotary-base ${ROPE_THETA}
  --rotary-scaling-factor ${SCALE_FACTOR}
  --swiglu
  --normalization RMSNorm
  --untie-embeddings-and-output-weights
  # 匹配
  # --cross-entropy-loss-fusion
)

MULTI_TOKEN_PREDICTION_ARGS=(
  # --use-multi-token-prediction
  # --mtp-coeff $MTP_COEFF
  # --mtp-depth $MTP_DEPTH
)

# 新增 TRACE_ARGS
TRACE_ARGS=(
  # --use-pytorch-profiler
  # --profile
  # --profile-ranks 0
)

if [ $GC = 1 ]; then
  GC_ARGS=(
    --manual-gc
    --manual-gc-interval 500
  )
else
  GC_ARGS=""
fi

if [ $OVERLAP = 1 ]; then
  OVERLAP_ARGS=(
    --overlap-grad-reduce
    --overlap-param-gather
  )
else
  OVERLAP_ARGS=""
fi

# 24414062 1T
TRAINING_ARGS=(
  --seed 1234
  --micro-batch-size $MICRO_BATCH_SIZE
  --global-batch-size $GLOBAL_BATCH_SIZE
  #  --rampup-batch-size 1200 1200 54931640
  #  --train-samples $TRAIN_SAMPLES
  --train-iters $TRAIN_ITERS
  --init-method-std 0.006 # 0.02 in HF config, but 0.006 in the paper
  --use-mcore-models
  # 匹配
  --no-bias-dropout-fusion
  --use-distributed-optimizer
  --use-flash-attn
  # 匹配
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers ${RECOMPUTE_LAYERS}
  --distributed-backend ${DIST_BACKEND}
  --multi-latent-attention
  --qk-layernorm
  "${LAST_STAGE_ARG}"
  # --mlp-recompute
  # --mlp-rms-recompute
  # --recompute-variance
  # --attn-recompute
  # --mla-rms-recompute
  --enable-experimental
  --no-rope-fusion
  --rotary-seq-len-interpolation-factor 1

  #  wlp
  --calculate-per-token-loss
  #  --end-weight-decay 0.0
  --override-opt_param-scheduler
  --use-rotary-position-embeddings
  --rerun-mode disabled
)

if [ $SFT = true ]; then
  TRAINING_ARGS+=(
    --dataset MMAP
    --train-mode finetune
    --finetune
    --no-load-optim
    --no-load-rng
    --dataloader-type cyclic
    --reset-position-ids
  )
fi

MLA_ARGS=(
  --kv-lora-rank ${KV_LORA_RANK}
  --qk-head-dim ${QK_NOPE_HEAD_DIM}
  --qk-pos-emb-head-dim ${QK_ROPE_HEAD_DIM}
  --v-head-dim ${V_HEAD_DIM}
  --kv-channels ${V_HEAD_DIM}
)

REGULARIZATION_ARGS=(
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.999
  --clip-grad 1.0
)

LEARNING_RATE_ARGS=(
  --lr 4.2e-05 #4.2e-4 wlp
  --lr-decay-style cosine
  --lr-wsd-decay-style exponential
  --lr-warmup-samples 0           # 128
  --min-lr 4.2e-06                # 4.2e-05
  --initial-loss-scale 4294967296 #65536 wlp
  --min-loss-scale 1.0
  --lr-warmup-iters 100 # wlp
)

MODEL_PARALLEL_ARGS=(
  --tensor-model-parallel-size $TP
  --pipeline-model-parallel-size $PP
  --tp-only-amax-red #对齐021 应注释该参数
)

if [ $PR = bf16 ]; then
  MIXED_PRECISION_ARGS=(
    --bf16
    #    --attention-softmax-in-fp32  wlp
    #    --no-masked-softmax-fusion wlp
    --accumulate-allreduce-grads-in-fp32
  )
fi

DATA_ARGS=(
  --data-path $DATA_PATH
  --tokenizer-type HuggingFaceTokenizer
  --tokenizer-model ${TOKENIZED_MODEL}
  --data-cache-path $DATA_CACHE_PATH
  --split 100,0,0
  --distributed-timeout-minutes 40
  --num-dataset-builder-threads 16
  --num-workers 2
)

if [ $FP8 = false ]; then
  TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl transformer_engine
  )
else
  TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl transformer_engine
    --fp8-format e4m3
    --fp8-recipe mxfp8
    --fp8-param-gather
  )
fi

NUM_LAYERS=$(echo "${MODEL_ARGS[@]}" | grep -oP '(?<=--num-layers )\d+')
NUM_LAYERS_MINUS_ONE=$((NUM_LAYERS - 0))
MOE_LAYER_FREQ="([0]*0+[1]*${NUM_LAYERS_MINUS_ONE})*1"

EVAL_AND_LOGGING_ARGS=(
  --log-interval 1
  --log-throughput
  --log-timers-to-tensorboard
  --save-interval $SAVE_INTERVAL
  --eval-interval 10000
  #  --save $CHECKPOINT_PATH
  --load $CHECKPOINT_LOAD_PATH
  --eval-iters 0
  --tensorboard-dir $TENSORBOARD_PATH
  --ckpt-format torch
  --exit-interval $EXIT_INTERVAL
  #  --logging-level 20
)
if [ $TIMER_PRINT = true ]; then
  MEGATRON_LOGGING_ARGS=(
    --timing-log-level 0
    --timing-log-option all
  )
else
  MEGATRON_LOGGING_ARGS=""
fi

MOE_ARGS=(
  --num-experts $NUM_EXPERTS
  --expert-model-parallel-size $EP
  --moe-token-dispatcher-type alltoall
  --moe-router-load-balancing-type aux_loss
  --moe-router-topk $ROUTER_TOPK
  --moe-router-score-function softmax
  --moe-router-topk-scaling-factor 2.446
  --moe-aux-loss-coeff 0.001
  --moe-ffn-hidden-size $MOE_INTERMEDIATE_SIZE #1536到768
  --moe-shared-expert-intermediate-size $((${MOE_INTERMEDIATE_SIZE} * ${NUM_SHARED_EXPERTS}))
  --moe-layer-freq $MOE_LAYER_FREQ
  --moe-grouped-gemm
  --moe-per-layer-logging
  # --moe-expert-capacity-factor 4
  # --moe-permute-fusion
  # --norm-before-router-softmax
  # --router-prob-var-mointor-freq 10
  # --router-logit-var-mointor-freq 10
  # --router-maxvio-mointor-freq 10
  --moe-router-pre-softmax
  --moe-z-loss-coeff 0.0001
)

if [ "$ROUTER_DTYPE" = "fp32" ] || [ "$ROUTER_DTYPE" = "fp64" ]; then
  MOE_ARGS+=(--moe-router-dtype "$ROUTER_DTYPE")
fi

###########################
###### running scripts
###########################

torchrun ${DISTRIBUTED_ARGS[@]} ${MEGATRON_MUSA_PATH}/examples/deepseek-sft/pretrain_deepseekv2.py \
  ${MODEL_ARGS[@]} \
  ${TRAINING_ARGS[@]} \
  ${REGULARIZATION_ARGS[@]} \
  ${LEARNING_RATE_ARGS[@]} \
  ${MODEL_PARALLEL_ARGS[@]} \
  ${MIXED_PRECISION_ARGS[@]} \
  ${DATA_ARGS[@]} \
  ${MOE_ARGS[@]} \
  ${MLA_ARGS[@]} \
  ${TRACE_ARGS[@]} \
  ${GC_ARGS[@]} \
  ${OVERLAP_ARGS[@]} \
  ${EVAL_AND_LOGGING_ARGS[@]} \
  ${MEGATRON_LOGGING_ARGS[@]} \
  ${MULTI_TOKEN_PREDICTION_ARGS[@]} \
  ${TRANSFORMER_ENGINE_ARGS[@]} 2>&1 | tee ${LOG_FILE}

set +x
