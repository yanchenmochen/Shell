#!/bin/bash

DATASET_DIR=/mnt/hw-nas/002147/cairunze/dataset/dataset
DATA_FORMAT=fp8
# 检查必要参数
if [[ -z "$DATASET_DIR" ]]; then
    echo "Error: --dataset_dir is required"
    exit 1
fi

# 生成唯一实验ID
CURRENT_TIME=$(date "+%Y%m%d_%H%M")
OUTPUT_FOLDER="${OUTPUT_DIR}/$CURRENT_TIME"
mkdir -p "${OUTPUT_FOLDER}"

# 训练配置 (使用Llama3 8B的并行参数)
TP_SIZE=${TP:-1}
PP_SIZE=${PP:-1}
EP_SIZE=1
MICRO_BATCH_SIZE=1

GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-1024}

# 设置环境变量
WORK_HOME="$PWD"
PATCH_HOME="$PWD/../.."
EXPNAME="tp${TP_SIZE}_pp${PP_SIZE}_mbs${MICRO_BATCH_SIZE}_gbs${GLOBAL_BATCH_SIZE}_gpus${WORLD_SIZE}_${DATA_FORMAT}_rank${RANK}"
DATA_PATH="${DATASET_DIR}/llama3-datasets/wudao_llama3bpe_content_document"
# HOSTFILE="./hostfile"
LOG_FILE="${OUTPUT_FOLDER}/$EXPNAME.log"
TOKENIZED_MODEL="${DATASET_DIR}/llama3_tokenizer"
SCRIPT_FILE="./8B/run_pretrain_llama3_musa_fp8.sh"
RDZV_ID="$CURRENT_TIME"


export OMP_NUM_THREADS=4
export MUSA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export MUSA_KERNEL_TIMEOUT=3200000
export ACCELERATOR_BACKEND="musa"
export MCCL_PROTOS=2
export MCCL_CHECK_POINTERS=0
export CUDA_DEVICE_MAX_CONNECTIONS=1

unset MLFLOW_TRACKING_URI

MEGATRON_PATH=${PATCH_HOME}/../Megatron-LM
export PYTHONPATH=${MEGATRON_PATH}:${PATCH_HOME}:$PYTHONPATH

if [ ! -d "${MEGATRON_PATH}/build" ]; then
    cd "${MEGATRON_PATH}"
    python setup.py build_ext --inplace
    cd -
fi

CHECKPOINT_PATH=$WORK_HOME/checkpoints/$EXPNAME
mkdir -p $CHECKPOINT_PATH
# DATA_PATH=$DATA_DIR


LOG_PATH=$WORK_HOME/logs/$EXPNAME
mkdir -p $LOG_PATH
cp $0 $LOG_PATH/
TB_PATH=$WORK_HOME/tboard/$EXPNAME
mkdir -p $TB_PATH
WB_PATH=$WORK_HOME/wandb/$EXPNAME
mkdir -p $WB_PATH

NUM_NODES=${WORLD_SIZE:-1}
CURRENT_TIME=$(date "+%Y-%m-%d_%H:%M")
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"7018"}
GPUS_PER_NODE=8
RECOMPUTE_LAYERS=0
NODE_RANK=${RANK:-0}

EXIT_INTERVAL=${EXIT_INTERVAL:-20000000}


DISTRIBUTED_ARGS=(
    --nproc_per_node 8 
    --nnodes $NUM_NODES 
    --node_rank $NODE_RANK 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT 
    --log_dir $WORK_HOME/output_log/$RDZV_ID/$EXPNAME
    --redirects ${LOG_REDIRECTS_LEVEL:-0}
)

MODEL_ARGS=(
    --num-layers 80
    --hidden-size 8192 
    --ffn-hidden-size 28672
    --num-attention-heads 64 
    --group-query-attention 
    --num-query-groups 8
    --seq-length 4096 
    --max-position-embeddings 131072 
    --norm-epsilon 1e-5 
    --attention-dropout 0.0 
    --hidden-dropout 0.0 
    --disable-bias-linear 
    --position-embedding-type rope 
    --no-position-embedding 
    --swiglu 
    --normalization RMSNorm
    --untie-embeddings-and-output-weights
)

# 244140625 1T
TRAINING_ARGS=(
    --seed 42 
    --micro-batch-size $MICRO_BATCH_SIZE 
    --global-batch-size $GLOBAL_BATCH_SIZE  
    --train-samples 24414062 
    --init-method-std 0.008
    --use-mcore-models 
    --no-bias-dropout-fusion
    --no-bias-swiglu-fusion
    --use-distributed-optimizer 
    --use-flash-attn 
    --sequence-parallel 
    --recompute-granularity full 
    --recompute-method block 
    --recompute-num-layers 0 
    --distributed-backend nccl
)


REGULARIZATION_ARGS=(
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --clip-grad 1.0 
)


LEARNING_RATE_ARGS=(
    --lr 1.5e-5 
    --lr-decay-style cosine 
    --min-lr 1.5e-6 
    --initial-loss-scale 65536 
    --min-loss-scale 1.0 
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size $TP_SIZE  
	--pipeline-model-parallel-size $PP_SIZE
)

MIXED_PRECISION_ARGS=(
    --bf16 
    --attention-softmax-in-fp32 
    --no-masked-softmax-fusion 
    --accumulate-allreduce-grads-in-fp32
)

DATA_ARGS="
    --data-path $DATA_PATH \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZED_MODEL} \
    --split 1
"


EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --log-throughput
    --save-interval 200000 
    --eval-interval 1000 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 0
    --tensorboard-dir $TB_PATH 
    --exit-interval $EXIT_INTERVAL
)

TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl transformer_engine
    --fp8-format hybrid
    --fp8-param-gather
    # --num-layers-per-virtual-pipeline-stage 10
    # --num-virtual-stages-per-pipeline-rank 4
    # --overlap-grad-reduce
    # --overlap-param-gather
    # --tp-comm-overlap
)
    # --num-virtual-stages-per-pipeline-rank 4
# if [ -n "${WANDB_API_KEY}" ]; then
#     EVAL_AND_LOGGING_ARGS+=(
#         --wandb-project ${WANDB_PROJECT:-"Mixtral-Finetuning"}
#         --wandb-exp-name ${WANDB_NAME:-"Mixtral_8x7B"} 
#     )
# fi

cmd="torchrun ${DISTRIBUTED_ARGS[@]} $WORK_HOME/pretrain_gpt.py \
        ${MODEL_ARGS[@]} \
        ${TRAINING_ARGS[@]} \
        ${REGULARIZATION_ARGS[@]}
        ${LEARNING_RATE_ARGS[@]} \
        ${MODEL_PARALLEL_ARGS[@]} \
        ${MIXED_PRECISION_ARGS[@]}
        ${DATA_ARGS[@]} \
        ${EVAL_AND_LOGGING_ARGS[@]} \
        ${TRANSFORMER_ENGINE_ARGS[@]} 2>&1 | tee ${LOG_FILE}
    "
echo $cmd
eval $cmd

# torchrun ${DISTRIBUTED_ARGS[@]} $WORK_HOME/pretrain_gpt.py \
#         ${MODEL_ARGS[@]} \
#         ${TRAINING_ARGS[@]} \
#         ${REGULARIZATION_ARGS[@]}
#         ${LEARNING_RATE_ARGS[@]} \
#         ${MODEL_PARALLEL_ARGS[@]} \
#         ${MIXED_PRECISION_ARGS[@]}
#         ${DATA_ARGS[@]} \
#         ${EVAL_AND_LOGGING_ARGS[@]} \
#         ${TRANSFORMER_ENGINE_ARGS[@]} 2>&1 | tee ${LOG_FILE}

# set +x
