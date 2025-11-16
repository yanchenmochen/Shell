#!/bin/bash
# This example will start serving the Llama3.1-8B model
# export NCCL_IB_SL=1
set -eo pipefail

export OMP_NUM_THREADS=4
export MUSA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export MUSA_KERNEL_TIMEOUT=3200000
export ACCELERATOR_BACKEND="musa"
export MCCL_PROTOS=2
export MCCL_CHECK_POINTERS=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
# export USE_RECOMPUTE_VARIANCE=1
# export NVTE_APPLY_QK_LAYER_SCALING=0
export MCCL_BUFFSIZE=20971520
export MUSA_BLOCK_SCHEDULE_MODE=1
export MCCL_IB_GID_INDEX=3
export MCCL_NET_SHARED_BUFFERS=0
export MCCL_IB_TC=122
export MCCL_IB_QPS_PER_CONNECTION=16
export PYTHONPATH=/mnt/seed-program-nas/001688/songquanheng/Shell/sft/megatron-lm-musa-patch_kuae_1022:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/openmpi/lib:/usr/local/musa/lib:${LD_LIBRARY_PATH}
export PYTORCH_SDP_BACKEND=math

# HOSTFILE=hostfile_sqh
# cat $HOSTFILE
# export NODE_ADDR=$(ip a|grep inet|grep -v 127.0.0.1|grep -v inet6|awk '{print $2;}'|tr -d "addr:"|head -n1 | cut -d '/' -f1) # tail for cuda/ head for musa
export GPUS_PER_NODE=8
export NUM_NODES=$(echo $WORLD_SIZE)
export MASTER_ADDR=$(echo $MASTER_ADDR)
export NODE_RANK=$(echo $POD_RANK)
export MASTER_PORT=12356

  

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --node_rank $NODE_RANK 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

echo ${DISTRIBUTED_ARGS[@]}

# Ensure CHECKPOINT and TOKENIZER_MODEL are provided
# if [ -z "$1" ] || [ -z "$2" ]; then
#   echo "Error: You must provide CHECKPOINT and TOKENIZER_MODEL as command-line arguments."
#   echo "Usage: $0 /path/to/checkpoint /path/to/tokenizer_model"
#   exit 1
# fi

TOKENIZER_MODEL='/mnt/seed-program-nas/001688/sft_ckpt/DeepSeek-V2-Lite-to-mcore-tp1-pp1-ep8-0928-x10000'
# 
CHECKPOINT='/mnt/seed-program-nas/001688/sft_ckpt/DeepSeek-V2-Lite-to-mcore-tp1-pp1-ep8-0928-x10000'

# TOKENIZER_MODEL='/mnt/seed-program-nas/001688/songquanheng/model/iter_0050000_hf_new'
# CHECKPOINT='/mnt/seed-program-nas/001688/songquanheng/model/iter_0050000_hf_new'

# pip install flask-restful
MODEL_SIZE=A2.4B
SEQ_LEN=4096
SEQ_LEN=4096
PR=bf16

TP=1
PP=1
EP=8


if [ $MODEL_SIZE = A2.4B ]; then

HIDDEN_SIZE=2048
NUM_ATTN_HEADS=16
NUM_LAYERS=27
INTERMEDIATE_SIZE=10944
MOE_INTERMEDIATE_SIZE=1408
MAX_POSITION_EMBEDDINGS=${SEQ_LEN}
EXTRA_VOCAB_SIZE=2400
KV_LORA_RANK=512
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


NUM_LAYERS_MINUS_ONE=$((NUM_LAYERS - 1))
MOE_LAYER_FREQ="([0]*1+[1]*${NUM_LAYERS_MINUS_ONE})*1"

moe_options=" \
    --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
    --moe-router-topk ${ROUTER_TOPK} \
    --num-experts ${NUM_EXPERTS} \
    --moe-layer-freq ${MOE_LAYER_FREQ} \
    --moe-aux-loss-coeff 1e-2 \
    --moe-shared-expert-intermediate-size $((${MOE_INTERMEDIATE_SIZE} * ${NUM_SHARED_EXPERTS} )) \
    --expert-model-parallel-size ${EP} \
    --kv-lora-rank ${KV_LORA_RANK} \
    --qk-head-dim ${QK_NOPE_HEAD_DIM} \
    --qk-pos-emb-head-dim ${QK_ROPE_HEAD_DIM} \
    --v-head-dim ${V_HEAD_DIM} \
    --moe-router-load-balancing-type aux_loss 
    " 
    

elif [ $MODEL_SIZE = A21B ]; then

HIDDEN_SIZE=5120
NUM_ATTN_HEADS=128
NUM_LAYERS=60
INTERMEDIATE_SIZE=12288
MOE_INTERMEDIATE_SIZE=1536
MAX_POSITION_EMBEDDINGS=${SEQ_LEN}
EXTRA_VOCAB_SIZE=2400
Q_LORA_RANK=1536
KV_LORA_RANK=512
QK_NOPE_HEAD_DIM=128
QK_ROPE_HEAD_DIM=64
V_HEAD_DIM=128
ROPE_THETA=10000
SCALE_FACTOR=40
NUM_EXPERTS=160
ROUTER_TOPK=6
NUM_SHARED_EXPERTS=2
MOE_LAYER_FREQ=1
RMS_NORM_EPS=1e-6

moe_options=" \
    --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
    --moe-router-topk ${ROUTER_TOPK} \
    --num-experts ${NUM_EXPERTS} \
    --moe-layer-freq ${MOE_LAYER_FREQ} \
    --moe-aux-loss-coeff 1e-2 \
    --moe-shared-expert-intermediate-size $((${MOE_INTERMEDIATE_SIZE} * ${NUM_SHARED_EXPERTS} )) \
    --expert-model-parallel-size ${EP} \
    --q-lora-rank ${Q_LORA_RANK} \
    --kv-lora-rank ${KV_LORA_RANK} \
    --qk-head-dim ${QK_NOPE_HEAD_DIM} \
    --qk-pos-emb-head-dim ${QK_ROPE_HEAD_DIM} \
    --v-head-dim ${V_HEAD_DIM} \
    --moe-router-load-balancing-type aux_loss"

fi

PR=bf16
if [ $PR = fp16 ]; then
    pr_options=" \
		    --fp16 \
            --apply-query-key-layer-scaling"
    export NVTE_APPLY_QK_LAYER_SCALING=1
elif [ $PR = bf16 ]; then
    pr_options=" \
        --bf16"
elif [ $PR = fp8 ]; then
    pr_options=" \
        --bf16 \
        --fp8-format hybrid \
        --fp8-amax-compute-algo max 
    "
fi

megatron_options="  \
        --tokenizer-type HuggingFaceTokenizer \
        --tokenizer-model $TOKENIZER_MODEL \
        --weight-decay 0.1 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --clip-grad 1.0 \
        --init-method-std 0.008 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --micro-batch-size 1 \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --ffn-hidden-size ${INTERMEDIATE_SIZE} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
        --log-interval 1 \
        --log-throughput \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --expert-model-parallel-size ${EP} \
        --context-parallel-size 1 \
        --no-load-optim \
        --no-load-rng \
        --num-workers 8 \
        --swiglu \
        --normalization RMSNorm \
        --norm-epsilon ${RMS_NORM_EPS} \
        --use-rotary-position-embeddings \
        --no-bias-swiglu-fusion \
        --no-rope-fusion \
        --position-embedding-type rope \
        --untie-embeddings-and-output-weights \
        --disable-bias-linear \
        --rotary-base ${ROPE_THETA} \
        --rotary-scaling-factor ${SCALE_FACTOR} \
        --no-save-optim \
        --kv-channels ${V_HEAD_DIM} \
        --qk-layernorm \
        --multi-latent-attention \
        --ckpt-format torch \
        --vocab-size 102400 \
        --make-vocab-size-divisible-by 102400 \
        --load ${CHECKPOINT} \
        --distributed-timeout-minutes 1000
        --model-name deepseek-v2-lite
        "
#  --make-vocab-size-divisible-by 128         

te_options=" \
        --transformer-impl transformer_engine"
        

cd /mnt/seed-program-nas/001688/songquanheng/Shell/sft/Megatron-LM_kuae_1022
cmd="torchrun ${DISTRIBUTED_ARGS[@]} tools/run_text_generation_server.py ${megatron_options} ${pr_options} ${moe_options}"
echo $cmd
$cmd