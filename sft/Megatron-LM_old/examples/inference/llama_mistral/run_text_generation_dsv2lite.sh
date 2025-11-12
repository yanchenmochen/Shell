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

DISTRIBUTED_ARGS="--nproc_per_node 1 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr 0.0.0.0 \
                  --master_port 6000"

# Ensure CHECKPOINT and TOKENIZER_MODEL are provided
# if [ -z "$1" ] || [ -z "$2" ]; then
#   echo "Error: You must provide CHECKPOINT and TOKENIZER_MODEL as command-line arguments."
#   echo "Usage: $0 /path/to/checkpoint /path/to/tokenizer_model"
#   exit 1
# fi

# Assign command-line arguments to variables
CHECKPOINT=/mnt/seed-program-nas/001688/sft_ckpt/DeepSeek-V2-Lite-to-mcore-tp1-pp1-ep8-0920
TOKENIZER_MODEL=/mnt/common/public/model/DeepSeek-V2-Lite

# pip install flask-restful
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

NUM_LAYERS_MINUS_ONE=$((NUM_LAYERS - 1))
MOE_LAYER_FREQ="([0]*1+[1]*${NUM_LAYERS_MINUS_ONE})*1"
MOE_FIRST_K_DENSE_REPLACE=1
RMS_NORM_EPS=1e-6

torchrun $DISTRIBUTED_ARGS tools/run_text_generation_server.py   \
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
      --rotary-base 500000 \
      --use-rope-scaling \
      --use-rotary-position-embeddings \
      --swiglu \
      --tensor-model-parallel-size 1  \
      --pipeline-model-parallel-size 1  \
      --expert-model-parallel-size 1 \
      --num-layers 8  \
      --hidden-size 2048  \
      --ffn-hidden-size $INTERMEDIATE_SIZE \
      --load ${CHECKPOINT}  \
      --num-attention-heads 16  \
      --max-position-embeddings $MAX_POSITION_EMBEDDINGS  \
      --bf16  \
      --micro-batch-size 1  \
      --multi-latent-attention \
      --qk-layernorm \
      --q-lora-rank ${Q_LORA_RANK} \
    --kv-lora-rank ${KV_LORA_RANK} \
    --qk-head-dim ${QK_NOPE_HEAD_DIM} \
    --qk-pos-emb-head-dim ${QK_ROPE_HEAD_DIM} \
    --v-head-dim ${V_HEAD_DIM} \
    --num-experts $NUM_EXPERTS \
    --moe-router-topk $ROUTER_TOPK \
    --moe-router-score-function softmax \
    --moe-router-topk-scaling-factor 2.643 \
    --moe-ffn-hidden-size $MOE_INTERMEDIATE_SIZE \
    --moe-shared-expert-intermediate-size $((${MOE_INTERMEDIATE_SIZE} * ${NUM_SHARED_EXPERTS} )) \
    --moe-layer-freq $MOE_LAYER_FREQ \
    --no-bias-dropout-fusion \
    --no-bias-swiglu-fusion \
    --use-distributed-optimizer \
    --moe-grouped-gemm \
    --moe-permute-fusion \
    --attention-backend unfused \
    --moe-token-dispatcher-type alltoall \
    --make-vocab-size-divisible-by 128 \
    --seq-length 4096
#    --use-flash-attn \