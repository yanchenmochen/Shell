#!/bin/bash

# 目标目录
TARGET_DIR="/mnt/seed-program-nas/001688/songquanheng/model/iter_0050000_hf_new"
# TARGET_DIR="/mnt/seed-program-nas/001688/honghong/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/moe32b/iter_0025000_hf_new"

# 拷贝本目录下的modeling_deepseek.py到目标目录
cp ./modeling_deepseek.py "$TARGET_DIR/"
cp ./config.json "$TARGET_DIR/"