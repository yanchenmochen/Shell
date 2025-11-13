#!/bin/bash
set -eo pipefail
set -x

TP=${TP:-1}
PP=${PP:-1}
EP=${EP:-1}
DIST_BACKEND=${DIST_BACKEND:-nccl}
PR=${PR:-bf16}
FP8=${FP8:-true}
GC=${GC:-1}
OVERLAP=${OVERLAP:-0}
TIMER_PRINT={TIMER_PRINT:-false}
TRAIN_ITERS=${TRAIN_ITERS:-100}
CHECKPOINT_LOAD_PATH=${CHECKPOINT_LOAD_PATH:-""}

echo "Using MUSA backend"

MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
#GLOBAL_BATCH_SIZE=4096


SEQ_LEN=${SEQ_LEN:-4096}
# DATASET_FILE=/mnt/seed-program-nas/001688/xiechunhong/1B_Loss_Align/dataset/datalist
# DATA_PATH="$(grep -v '^#' ${DATASET_FILE})"
# DATA_CACHE_PATH=/mnt/seed-program-nas/001688/xiechunhong/1B_Loss_Align/dataset/datacache
TOKENIZED_MODEL=/mnt/seed-program-nas/001688/xiechunhong/1B_Loss_Align/Meta-Llama-3-tokenizer
MODEL_NAME=${MODEL_NAME:-'MOE-32B'}

if [[ -z ${SEEN_STEPS} ]]; then
   SEEN_STEPS=0
fi
# SAMPLE_SIZE="$(($(python sum_column_1.py --datalist_file ${DATASET_FILE})/${SEQ_LEN}))"

# SAMPLE_ITERS="$((${SAMPLE_SIZE}/${GLOBAL_BATCH_SIZE}))"
# TRAIN_STEPS=$((${SEEN_STEPS} + ${SAMPLE_ITERS}))
# TOTAL_STEPS=$TRAIN_STEPS

# TRAIN_SAMPLES=$SAMPLE_SIZE

# TOTAL_TOKENS=12883845026336

DATASET_PATH="$(grep -v '^#' ${DATASET_FILE})"
VALID_DATASET_PATH=${DATASET_PATH}

TOTAL_TOKENS=8106518565204
# TOTAL_TOKENS=5306132903dd
#TOTAL_TOKENS=1000000000000
SAMPLE_SIZE="$((${TOTAL_TOKENS}/${SEQ_LEN}))"
TRAIN_SAMPLES=$SAMPLE_SIZE
TRAIN_ITERS=$(( ${TOTAL_TOKENS} / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))

WARMUP_TOKENS=$(( ${TOTAL_TOKENS} / 100 ))
LR_WARMUP_SAMPLES=$(( ${WARMUP_TOKENS} / ${SEQ_LEN} ))
LR_WARMUP_ITERS=$(( ${WARMUP_TOKENS} / ${SEQ_LEN} / ${GLOBAL_BATCH_SIZE} ))

WARMUP_STEPS=2000
WARMUP_SAMPLES=$((WARMUP_STEPS * GLOBAL_BATCH_SIZE))

OUTPUT_DIR=${OUTPUT_DIR:-"./output"}


unset MLFLOW_TRACKING_URI

###########################
###### change for multinode config
###########################
NUM_NODES=${WORLD_SIZE:-1}
CURRENT_TIME=$(date "+%Y-%m-%d_%H:%M")
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"7018"}
GPUS_PER_NODE=${TQ_GPU_NUM:-8}
RECOMPUTE_LAYERS=0
WORLD_SIZE=$((GPUS_PER_NODE * NUM_NODES))
DP_SIZE=$((WORLD_SIZE / (PP * TP)))
NODE_RANK=${RANK:-0}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
EXIT_INTERVAL=${EXIT_INTERVAL:-20000000}
###########################
EXPNAME="zj_tp${TP}_pp${PP}_dp${DP_SIZE}_ep${EP}_mbs${MICRO_BATCH_SIZE}_numbs${NUM_MICROBATCHES}_gbs${GLOBAL_BATCH_SIZE}_gpus${WORLD_SIZE}"

###########################
###### envrioment variables
###########################
# export ENABLE_D2H_IN_PERMUTATION=1  #摩尔脚本无
export NO_LOSS_REDUCE=1
export USE_RECOMPUTE_VARIANCE=1
# export USE_MUSA_MOE=1

export LOGLEVEL="INFO"
export MUSA_EXECUTION_TIMEOUT=20000000
export MUSA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export CUDA_DEVICE_MAX_CONNECTIONS=1
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
export MCCL_IB_TC=122
export MCCL_IB_QPS_PER_CONNECTION=16


###########################
###### commonly used args
###########################
# MICRO_BS=1
# MICRO_CNT=64
# PP_SIZE=1
# TP_SIZE=1
# EP_SIZE=8
# HIDDEN_SIZE=2048
# NUM_LAYERS=48 # 61 infact

# 021-32B最新参数
HIDDEN_SIZE=2048
NUM_ATTN_HEADS=16
# 021-32B 为40层
NUM_LAYERS=8
INTERMEDIATE_SIZE=10944
MOE_INTERMEDIATE_SIZE=1408
MAX_POSITION_EMBEDDINGS=4096
Q_LORA_RANK=1536
KV_LORA_RANK=512
QK_NOPE_HEAD_DIM=128
QK_ROPE_HEAD_DIM=64
V_HEAD_DIM=128
ROPE_THETA=10000
SCALE_FACTOR=1
NUM_EXPERTS=64
ROUTER_TOPK=6
NUM_SHARED_EXPERTS=2
MOE_LAYER_FREQ=1
MOE_FIRST_K_DENSE_REPLACE=1
RMS_NORM_EPS=1e-6


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

###########################
###### exp name and log dir
###########################


echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "NODE_ADDR: ${NODE_ADDR}"
echo "NODE_RANK: ${NODE_RANK}"

mkdir -p ${OUTPUT_DIR}/logs/${CURRENT_TIME}
LOG_FILE_TMP="${OUTPUT_DIR}/logs/${CURRENT_TIME}/${EXPNAME}.${NODE_RANK}.${NODE_ADDR}.log"
LOG_FILE=${LOG_FILE:-${LOG_FILE_TMP}}
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
    # --log_dir $WORK_HOME/output/output_log/$RDZV_ID/$EXPNAME
    # --redirects 1
)

MODEL_ARGS=(
    --num-layers $NUM_LAYERS  # 60
    --hidden-size ${HIDDEN_SIZE}
    --num-attention-heads $NUM_ATTN_HEADS
    --seq-length ${SEQ_LEN}
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS
    --norm-epsilon $RMS_NORM_EPS
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --disable-bias-linear
    # --vocab-size 102400
    --ffn-hidden-size $INTERMEDIATE_SIZE
    --position-embedding-type rope
    # --use-rotary-position-embeddings
    #--no-position-embedding
    # 匹配
    --rotary-base ${ROPE_THETA}
    --rotary-scaling-factor ${SCALE_FACTOR}
    # --no-position-embedding
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
    --seed 42
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    # --train-samples 24414062
    # --train-iters ${TRAIN_ITERS}
    --train-samples $TRAIN_SAMPLES
    --init-method-std  0.006 # 0.02 in HF config, but 0.006 in the paper
    --use-mcore-models
    # 匹配
    # --no-gradient-accumulation-fusion
    --no-bias-dropout-fusion
    # --no-bias-swiglu-fusion
    --use-distributed-optimizer
    --use-flash-attn
    # 匹配
    # --sequence-parallel
    --recompute-granularity full
    --recompute-method block
    --recompute-num-layers ${RECOMPUTE_LAYERS}
    --distributed-backend ${DIST_BACKEND}
    --multi-latent-attention
    --qk-layernorm

    "${LAST_STAGE_ARG}"

    --mlp-recompute
    --mlp-rms-recompute
    --recompute-variance
)

MLA_ARGS=(
    --q-lora-rank ${Q_LORA_RANK}
    --kv-lora-rank ${KV_LORA_RANK}
    --qk-head-dim ${QK_NOPE_HEAD_DIM}
    --qk-pos-emb-head-dim ${QK_ROPE_HEAD_DIM}
    --v-head-dim ${V_HEAD_DIM}
)

REGULARIZATION_ARGS=(
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --clip-grad 1.0
)



LEARNING_RATE_ARGS=(
    --lr 3.2e-4
    --lr-decay-style WSD
    # --lr-warmup-samples ${WARMUP_SAMPLES}
    --lr-wsd-decay-samples  $((TRAIN_SAMPLES * 2 /10))
    --lr-wsd-decay-style cosine
    --lr-warmup-samples ${WARMUP_SAMPLES}
    --min-lr 3.2e-05
    --initial-loss-scale 65536
    --min-loss-scale 1.0
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size $TP
	--pipeline-model-parallel-size $PP

    # 对齐复现
    --tp-only-amax-red #对齐021 应注释该参数
    # --use-tp-pp-dp-mapping
)

if [ $PR = bf16 ]; then
    MIXED_PRECISION_ARGS=(
        --bf16
        --attention-softmax-in-fp32
        --no-masked-softmax-fusion
        --accumulate-allreduce-grads-in-fp32
        # 匹配
        # --grad-reduce-in-bf16
    )
fi

DATA_ARGS=(
    # --tokenizer-type NullTokenizer
    # # --tokenizer-model ${TOKENIZED_MODEL}
    --data-path $DATA_PATH
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model ${TOKENIZED_MODEL}
    --data-cache-path $DATA_CACHE_PATH
    --split 100,0,0
    --distributed-timeout-minutes 1000
)

# TRANSFORMER_ENGINE_ARGS=(
#     --transformer-impl transformer_engine
#     # --transformer-impl local
#     # --fp8-format hybrid
#     # --fp8-param-gather
# )
if [ $FP8 = false ]; then
    TRANSFORMER_ENGINE_ARGS=(
        --transformer-impl transformer_engine
        # # --transformer-impl local
        # --fp8-format hybrid
        # --fp8-param-gather
        # 关掉fp8容易显存占满，增加recompute操作
        --attn-recompute
        --mla-rms-recompute #@huang
    )
else
    TRANSFORMER_ENGINE_ARGS=(
        --transformer-impl transformer_engine
        # # --transformer-impl local
        --fp8-format hybrid
        --fp8-param-gather
    )
fi



NUM_LAYERS=$(echo "${MODEL_ARGS[@]}" | grep -oP '(?<=--num-layers )\d+')
NUM_LAYERS_MINUS_ONE=$((NUM_LAYERS - 1))
MOE_LAYER_FREQ="([0]*1+[1]*${NUM_LAYERS_MINUS_ONE})*1"


EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --log-throughput
    --log-timers-to-tensorboard
    --log-params-norm
    --log-num-zeros-in-grad
    --save-interval $SAVE_INTERVAL
    --eval-interval 1
    --save $CHECKPOINT_PATH
    # --load $CHECKPOINT_LOAD_PATH
    --eval-iters 0
    --tensorboard-dir $TENSORBOARD_PATH
    --ckpt-format torch
    --exit-interval $EXIT_INTERVAL
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
    # 匹配
    # --moe-router-num-groups $EP_SIZE
    # --moe-router-group-topk 1
    --moe-router-load-balancing-type aux_loss

    #匹配
    # --moe-router-load-balancing-type seq_aux_loss
    --moe-router-topk $ROUTER_TOPK
    # --moe-router-pre-softmax #deepseek use pre-softmax
    --moe-router-score-function softmax
    # --moe-router-norm-topk-prob
    --moe-router-topk-scaling-factor 2.643 # pre-softmax need scaling
    --moe-aux-loss-coeff 0.001
    # 匹配
    # --moe-expert-capacity-factor 1
    # --moe-device-level-capacity
    # --moe-device-level-aux-loss-coeff 5e-2
    # --moe-comm-aux-loss-coeff 2e-2
    --moe-ffn-hidden-size $MOE_INTERMEDIATE_SIZE  #1536到768
    --moe-shared-expert-intermediate-size $((${MOE_INTERMEDIATE_SIZE} * ${NUM_SHARED_EXPERTS} ))
    --moe-layer-freq $MOE_LAYER_FREQ
    --moe-grouped-gemm
    --moe-permute-fusion



    # 为复现moer 0630的结果设置
    --moe-router-norm-topk-prob #对齐应注释
    # --moe-router-load-balancing-type seq_aux_loss # 对齐应为aux_loss
    # --moe-expert-capacity-factor 4.0 # 对齐应注释掉
)


# --moe-z-loss-coeff 1e-3
# --moe-expert-capacity-factor 4.0
# --moe-pad-expert-input-to-capacity
# if [ -n "${WANDB_API_KEY}" ]; then
#     EVAL_AND_LOGGING_ARGS+=(
#         --wandb-project ${WANDB_PROJECT:-"Mixtral-Finetuning"}
#         --wandb-exp-name ${WANDB_NAME:-"Mixtral_8x7B"}
#     )
# fi

###########################
###### running scripts
###########################


ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )
ROOT_DIR=$(realpath "${ROOT_DIR}/../..")
echo $ROOT_DIR
cur_dir=$( pwd )
cd /mnt/seed-program-nas/001688/xiechunhong/1B_Loss_Align/Megatron-LM
python setup.py build_ext --inplace
cd $cur_dir


# cat "${ROOT_DIR}/version.txt"


MEGATRON_PATH=${ROOT_DIR}
export PYTHONPATH=/mnt/seed-program-nas/001688/haoran.huang/megatron-lm-musa-patch:/mnt/seed-program-nas/001688/xiechunhong/1B_Loss_Align/Megatron-LM:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/openmpi/lib:/usr/local/musa/lib:${LD_LIBRARY_PATH}

# run_cmd="torchrun ${DISTRIBUTED_ARGS[@]} pretrain_deepseekv2.py \
#         ${MODEL_ARGS[@]} \
#         ${TRAINING_ARGS[@]} \
#         ${REGULARIZATION_ARGS[@]} \
#         ${LEARNING_RATE_ARGS[@]} \
#         ${MODEL_PARALLEL_ARGS[@]} \
#         ${MIXED_PRECISION_ARGS[@]} \
#         ${DATA_ARGS[@]} \
#         ${MOE_ARGS[@]} \
#         ${MLA_ARGS[@]} \
#         ${TRACE_ARGS[@]} \
#         ${GC_ARGS[@]} \
#         ${OVERLAP_ARGS[@]} \
#         ${EVAL_AND_LOGGING_ARGS[@]} \
#         ${MEGATRON_LOGGING_ARGS[@]} \
#         ${MULTI_TOKEN_PREDICTION_ARGS[@]} \
#         ${TRANSFORMER_ENGINE_ARGS[@]} 2>&1 | tee ${LOG_FILE}
#  "
# echo ${run_cmd}
# eval ${run_cmd}
# set +x

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_deepseekv2.py \
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
