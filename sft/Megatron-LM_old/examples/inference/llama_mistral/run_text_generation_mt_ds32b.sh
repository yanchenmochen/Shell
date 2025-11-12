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
export USE_RECOMPUTE_VARIANCE=1
# export NVTE_APPLY_QK_LAYER_SCALING=0
export MCCL_BUFFSIZE=20971520
export MUSA_BLOCK_SCHEDULE_MODE=1
export MCCL_IB_GID_INDEX=3
export MCCL_NET_SHARED_BUFFERS=0
export MCCL_IB_TC=122
export MCCL_IB_QPS_PER_CONNECTION=16
export PYTHONPATH=/mnt/seed-program-nas/001688/haoran.huang/megatron-lm-musa-patch:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/openmpi/lib:/usr/local/musa/lib:${LD_LIBRARY_PATH}
export PYTORCH_SDP_BACKEND=math

HOSTFILE=/mnt/seed17/001688/haoran.huang/Megatron-LM_old/examples/inference/llama_mistral/hostfile
cat $HOSTFILE
export NODE_ADDR=$(ip a|grep inet|grep -v 127.0.0.1|grep -v inet6|awk '{print $2;}'|tr -d "addr:"|head -n1 | cut -d '/' -f1) # tail for cuda/ head for musa
export GPUS_PER_NODE=8
export NUM_NODES=$(cat $HOSTFILE | wc -l)
export MASTER_ADDR=$(head -n1 $HOSTFILE | awk '{print $1;}')
export NODE_RANK=$(awk -v node_addr="$NODE_ADDR" '{ranks[$1]=(FNR-1);} END {print ranks[node_addr];}' $HOSTFILE)
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

TOKENIZER_MODEL='/mnt/seed-program-nas/001688/xiechunhong/1B_Loss_Align/Meta-Llama-3-tokenizer'
CHECKPOINT='/mnt/seed-program-nas/001688/checkpoint/021_32B'
CHECKPOINT='/mnt/seed17/001688/haoran.huang/megatron-lm-musa-patch/examples/deepseek-sft/checkpoints/tp1_pp2_dp8_mbs1_numbs128_gbs1024_gpus16'

# pip install flask-restful
SEQ_LEN=4096
PR=bf16

TP=1
PP=2
EP=8

HIDDEN_SIZE=2048
NUM_ATTENTION_HEADS=32
NUM_LAYERS=40
INTERMEDIATE_SIZE=12288
MOE_INTERMEDIATE_SIZE=1536
# EXTRA_VOCAB_SIZE=2400 # 021
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
MOE_FIRST_K_DENSE_REPLACE=1 # 021
RMS_NORM_EPS=1e-6

NUM_LAYERS_MINUS_ONE=$((NUM_LAYERS - 1))
MOE_LAYER_FREQ="([0]*1+[1]*${NUM_LAYERS_MINUS_ONE})*1"


# --max-padding-length 4096 --extra-vocab-size 0 --enable-shared-expert --qk-nope-head-dim 128 --qk-rope-head-dim 64 --moe-router-norm-topk-prob     --attention-backend unfused \
cd /mnt/seed17/001688/haoran.huang/Megatron-LM_old
cmd="torchrun ${DISTRIBUTED_ARGS[@]} tools/run_text_generation_server.py  \
    --ckpt-format torch \
    --use-mcore-models \
    --disable-bias-linear \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --transformer-impl transformer_engine \
    --normalization RMSNorm \
    --attention-softmax-in-fp32 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --untie-embeddings-and-output-weights \
    --position-embedding-type rope \
    --rotary-base ${ROPE_THETA} \
    --use-rotary-position-embeddings \
    --swiglu \
    --tensor-model-parallel-size $TP  \
    --pipeline-model-parallel-size $PP  \
    --expert-model-parallel-size ${EP} \
    --num-layers ${NUM_LAYERS}  \
    --hidden-size ${HIDDEN_SIZE} \
    --ffn-hidden-size ${INTERMEDIATE_SIZE} \
    --load ${CHECKPOINT} \
    --num-attention-heads ${NUM_ATTENTION_HEADS} \
    --max-position-embeddings 4096  \
    --bf16 \
    --micro-batch-size 1  \
    --multi-latent-attention \
    --qk-layernorm \
    --q-lora-rank ${Q_LORA_RANK} \
    --kv-lora-rank ${KV_LORA_RANK} \
    --qk-head-dim ${QK_NOPE_HEAD_DIM} \
    --qk-pos-emb-head-dim ${QK_ROPE_HEAD_DIM} \
    --v-head-dim ${V_HEAD_DIM} \
    --num-experts ${NUM_EXPERTS} \
    --moe-router-topk ${ROUTER_TOPK} \
    --moe-router-score-function softmax \
    --moe-router-topk-scaling-factor 2.643 \
    --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
    --moe-shared-expert-intermediate-size $((${MOE_INTERMEDIATE_SIZE} * ${NUM_SHARED_EXPERTS} )) \
    --moe-layer-freq $MOE_LAYER_FREQ \
    --no-bias-swiglu-fusion \
    --moe-grouped-gemm \
    --moe-token-dispatcher-type alltoall \
    --make-vocab-size-divisible-by 128 \
    --seq-length 4096 
    --no-rope-fusion \
    --moe-router-load-balancing-type aux_loss \
    --moe-aux-loss-coeff 1e-2 \
    --no-async-tensor-model-parallel-allreduce \
    --context-parallel-size 1 \
    --num-workers 2 \
    --norm-epsilon ${RMS_NORM_EPS} \
    --rotary-scaling-factor ${SCALE_FACTOR} \
    --rotary-seq-len-interpolation-factor 1 \
    --kv-channels ${V_HEAD_DIM}
    "
echo $cmd
$cmd