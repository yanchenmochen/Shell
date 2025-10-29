#!/bin/bash

# 检查是否提供了 GPU 编号作为参数
if [ -z "$1" ]; then
  echo "No GPU_ID provided."
  echo "Current values of environment variables:"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
  echo "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-not set}"
  exit 0
fi

# 获取 GPU 编号
GPU_ID=$1

# 清除环境变量
unset CUDA_VISIBLE_DEVICES
unset NVIDIA_VISIBLE_DEVICES

# 设置新的环境变量
export CUDA_VISIBLE_DEVICES=$GPU_ID
export NVIDIA_VISIBLE_DEVICES=$GPU_ID

# 打印设置后的环境变量以供验证
echo "CUDA_VISIBLE_DEVICES set to $CUDA_VISIBLE_DEVICES"
echo "NVIDIA_VISIBLE_DEVICES set to $NVIDIA_VISIBLE_DEVICES"
