#!/bin/bash
set -eo pipefail
set -x

export PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True

TP=${TP:-1}
PP=${PP:-1}
EP=${EP:-8}
DIST_BACKEND=${DIST_BACKEND:-nccl}
PR=${PR:-bf16}
FP8=${FP8:-false}
GC=${GC:-1}
OVERLAP=${OVERLAP:-0}
TIMER_PRINT={TIMER_PRINT:-false}
TRAIN_ITERS=${TRAIN_ITERS:-100}
CHECKPOINT_LOAD_PATH=${CHECKPOINT_LOAD_PATH:-}

echo "Using MUSA backend"

ROUTER_DTYPE=${ROUTER_DTYPE:-fp32}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-1024}
#GLOBAL_BATCH_SIZE=4096

SEQ_LEN=${SEQ_LEN:-4096}
DATASET_FILE=/mnt/moer-train/public/datalist/021-128T-part1.datalist
DATA_PATH="$(grep -v '^#' ${DATASET_FILE})"
DATA_CACHE_PATH=/mnt/dataset-12.8t/public/datacache/data_128t_1111
TOKENIZED_MODEL=/mnt/hw-nas/public/models/zjllm-llama3-tokenizer
MODEL_NAME=${MODEL_NAME:-'021-32B-1024'}

if [[ -z ${SEEN_STEPS} ]]; then
   SEEN_STEPS=0
fi


#DATASET_PATH="$(grep -v '^#' ${DATASET_FILE})"
#VALID_DATASET_PATH=${DATASET_PATH}

# TOTAL_TOKENS=8106518565204
# # TOTAL_TOKENS=5306132903
# SAMPLE_SIZE="$((${TOTAL_TOKENS}/${SEQ_LEN}))"
# TRAIN_SAMPLES=$SAMPLE_SIZE
# TRAIN_ITERS=$(( ${TOTAL_TOKENS} / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))

# WARMUP_TOKENS=$(( ${TOTAL_TOKENS} / 100 ))
# LR_WARMUP_SAMPLES=$(( ${WARMUP_TOKENS} / ${SEQ_LEN} ))
# LR_WARMUP_ITERS=$(( ${WARMUP_TOKENS} / ${SEQ_LEN} / ${GLOBAL_BATCH_SIZE} ))

# WARMUP_STEPS=2000
# WARMUP_SAMPLES=$((WARMUP_STEPS * GLOBAL_BATCH_SIZE))

# DATASET_PATH="$(grep -v '^#' ${DATASET_FILE})"
# VALID_DATASET_PATH=${DATASET_PATH}

TOTAL_TOKENS=12883845026336
#TOTAL_TOKENS=9883132548256
# TOTAL_TOKENS=5306132903
SAMPLE_SIZE="$((${TOTAL_TOKENS}/${SEQ_LEN}))"
TRAIN_SAMPLES=$SAMPLE_SIZE
TRAIN_ITERS=$(( ${TOTAL_TOKENS} / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))

#WARMUP_TOKENS=$(( ${TOTAL_TOKENS} / 100 ))
#LR_WARMUP_SAMPLES=$(( ${WARMUP_TOKENS} / ${SEQ_LEN} ))
#LR_WARMUP_ITERS=$(( ${WARMUP_TOKENS} / ${SEQ_LEN} / ${GLOBAL_BATCH_SIZE} ))

WARMUP_STEPS=2000
WARMUP_SAMPLES=$((WARMUP_STEPS * 1600))


OUTPUT_DIR=${OUTPUT_DIR:-"./output"}


unset MLFLOW_TRACKING_URI
unset MCCL_IB_HCA


###########################
###### change for multinode config
###########################
NUM_NODES=${WORLD_SIZE:-1}
CURRENT_TIME=$(date "+%Y-%m-%d_%H:%M")
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
# MASTER_PORT=${MASTER_PORT:-"7018"}
MASTER_PORT=9958
GPUS_PER_NODE=${TQ_GPU_NUM:-8}
RECOMPUTE_LAYERS=1
WORLD_SIZE=$((GPUS_PER_NODE * NUM_NODES))
DP_SIZE=$((WORLD_SIZE / (PP * TP)))
NODE_RANK=${RANK:-0}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
EXIT_INTERVAL=${EXIT_INTERVAL:-20000000}
###########################
EXPNAME="zj_tp${TP}_pp${PP}_dp${DP_SIZE}_ep${EP}_mbs${MICRO_BATCH_SIZE}_numbs${NUM_MICROBATCHES}_gbs${GLOBAL_BATCH_SIZE}_gpus${WORLD_SIZE}_router${ROUTER_DTYPE}"

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
export MCCL_IB_TC=136
export MCCL_IB_QPS_PER_CONNECTION=16
export MCCL_CROSS_NIC=0
# export MCCL_IB_HCA=mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5

export MCCL_IB_TIMEOUT=20
export MCCL_IB_RETRY_CNT=7

export LOGLEVEL="INFO"
export MUSA_PRINT_ENV=1
export MUSA_EXECUTION_TIMEOUT=3000000
export MUSA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export ACCELERATOR_BACKEND="musa"
export MCCL_CHECK_POINTERS=0
export OMP_NUM_THREADS=4
export MUSA_BLOCK_SCHEDULE_MODE=1 
export MCCL_IB_QPS_PER_CONNECTION=16
export MUSA_BLOCK_DISTRIBUTION_GRANULARITY=0
export MCCL_CROSS_NIC=0
export MCCL_IB_TC=136
export MCCL_IB_TIMEOUT=20
export MCCL_IB_RETRY_CNT=7
export MUSA_LOG=0x1
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

if [ $MODEL_SIZE = A2.4B ]; then
    HIDDEN_SIZE=2048
    NUM_ATTN_HEADS=16
    NUM_LAYERS=27
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
    SCALE_FACTOR=40
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

###########################
###### exp name and log dir
###########################

# 获取当前机器IP
NODE_ADDR=$(ip a | awk '/inet / && !/127.0.0.1/ {print $2}' | cut -d/ -f1 | head -n 1)

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
    --mtp-loss-scaling-factor 0.3
    --mtp-num-layers 1
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
    --rampup-batch-size 1600 800 50000000
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
    # --recompute-granularity full 
    # --recompute-method uniform 
    # --recompute-num-layers ${RECOMPUTE_LAYERS}
    --distributed-backend ${DIST_BACKEND}
    --multi-latent-attention
    --qk-layernorm
    "${LAST_STAGE_ARG}"    
    --mlp-recompute
    --mlp-rms-recompute 
    --recompute-variance
    --attn-recompute
    # --mla-rms-recompute
    --enable-experimental
    # --no-rope-fusion
    --rotary-seq-len-interpolation-factor 1
)

MLA_ARGS=(
    # --q-lora-rank ${Q_LORA_RANK} 
    --kv-lora-rank ${KV_LORA_RANK} 
    --qk-head-dim ${QK_NOPE_HEAD_DIM} 
    --qk-pos-emb-head-dim ${QK_ROPE_HEAD_DIM} 
    --v-head-dim ${V_HEAD_DIM} 
    --kv-channels ${V_HEAD_DIM}
)

REGULARIZATION_ARGS=(
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --clip-grad 1.0 
)



LEARNING_RATE_ARGS=(
    --lr 3.2e-4
    --lr-decay-style cosine 
    # --lr-decay-samples 2854631639
    --lr-warmup-samples ${WARMUP_SAMPLES} 
    # --lr-wsd-decay-samples  $((TRAIN_SAMPLES * 2 /10))
    # --lr-wsd-decay-style cosine
    # --lr-warmup-samples ${WARMUP_SAMPLES}
    --min-lr 3.2e-05
    --initial-loss-scale 65536 
    --min-loss-scale 1.0 
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size $TP  
	--pipeline-model-parallel-size $PP 
    --tp-only-amax-red #对齐021 应注释该参数
)

if [ $PR = bf16 ]; then
    MIXED_PRECISION_ARGS=(
        --bf16 
        --attention-softmax-in-fp32 
        --no-masked-softmax-fusion 
        --accumulate-allreduce-grads-in-fp32
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
    --distributed-timeout-minutes 480
    --num-dataset-builder-threads 16
    --num-workers 2
    #--no-mmap-bin-files
)


if [ $FP8 = false ]; then
    TRANSFORMER_ENGINE_ARGS=(
        --transformer-impl transformer_engine
    )
else
    # TRANSFORMER_ENGINE_ARGS=(
    #     --transformer-impl transformer_engine
    #     # # --transformer-impl local
    #     --fp8-format hybrid
    #     --fp8-param-gather
    # )
    TRANSFORMER_ENGINE_ARGS=(
        --transformer-impl transformer_engine
        --fp8-format e4m3
        --fp8-recipe mxfp8
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
    # --log-params-norm
    # --log-num-zeros-in-grad
    --save-interval $SAVE_INTERVAL
    --eval-interval 1
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_LOAD_PATH
    --eval-iters 0
    --tensorboard-dir $TENSORBOARD_PATH
    --ckpt-format torch
    --exit-interval $EXIT_INTERVAL
    --logging-level 20
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
    --moe-router-topk-scaling-factor 2.643 
    --moe-aux-loss-coeff 0.001
    --moe-ffn-hidden-size $MOE_INTERMEDIATE_SIZE  #1536到768
    --moe-shared-expert-intermediate-size $((${MOE_INTERMEDIATE_SIZE} * ${NUM_SHARED_EXPERTS} ))
    --moe-layer-freq $MOE_LAYER_FREQ
    --moe-grouped-gemm
    --moe-per-layer-logging
    # --moe-expert-capacity-factor 4
    --moe-permute-fusion		
    # --norm-before-router-softmax
    --router-prob-var-mointor-freq 10
    --router-logit-var-mointor-freq 10
    --router-maxvio-mointor-freq 10
    --moe-router-pre-softmax 
    --moe-z-loss-coeff 0.0001
)


if [ "$ROUTER_DTYPE" = "fp32" ] || [ "$ROUTER_DTYPE" = "fp64" ]; then
    MOE_ARGS+=(--moe-router-dtype "$ROUTER_DTYPE")
fi
###########################
###### running scripts
###########################


ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )
ROOT_DIR=$(realpath "${ROOT_DIR}/../..") 
echo $ROOT_DIR
cur_dir=$( pwd )

#cd /home/Megatron-LM
#python setup.py build_ext --inplace
echo "cur dir: ${cur_dir}"
cd $cur_dir

echo "entering current dir $cur_dir"

export PYTHONPATH=/mnt/moer-train/public/train_32B/Kuae2.1-1022/megatron-lm-musa-patch:/mnt/moer-train/public/train_32B/Kuae2.1-1022/Megatron-LM:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/openmpi/lib:/usr/local/musa/lib:${LD_LIBRARY_PATH}

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
