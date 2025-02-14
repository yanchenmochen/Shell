#!/bin/bash

# Set default values for environment variables
export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-12345}"
export WORLD_SIZE="${WORLD_SIZE:-1}"
export RANK="${RANK:-0}"
export TQ_GPU_NUM="${TQ_GPU_NUM:-8}"
export megatron_home="${megatron_home:-/mnt/self-define/songquanheng/zjlab-megatron}"

# Output the current values of the environment variables
echo "MASTER_ADDR set to: $MASTER_ADDR"
echo "MASTER_PORT set to: $MASTER_PORT"
echo "NNODES set to: $NNODES"
echo "NODE_RANK set to: $NODE_RANK"
echo "GPUS_PER_NODE set to: $GPUS_PER_NODE"
echo "megatron_home set to: $megatron_home"


