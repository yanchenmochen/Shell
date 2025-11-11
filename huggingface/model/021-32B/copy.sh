#!/bin/bash
# 脚本当前可运行于 镜像10.200.53.208/001688/021-32b-inference:v1
# 脚本主要的目的，主要在修改modeling_deepseek.py 是将config.json、modeling_deepseek.py 拷贝到模型目录下，
# ，然后调用python $shell_home/huggingface/model_generator_021.py 可以载入这个。
# 在运行时，会将modeling_deepseek.py 拷贝到cache目录下 ~/.cache/huggingface/modules/transformers_modules
# 想要断点的话，可以打开该文件添加，实现断点调试的功能

# 目标目录
TARGET_DIR="/mnt/seed-program-nas/001688/songquanheng/model/iter_0050000_hf_new"
TARGET_DIR="/mnt/seed-program-nas/001688/honghong/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/moe32b/iter_0030000_hf_1107"
# TARGET_DIR="/mnt/seed-program-nas/001688/honghong/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/moe32b/iter_0025000_hf_new"

# 拷贝本目录下的modeling_deepseek.py到目标目录
cp ./modeling_deepseek.py "$TARGET_DIR/"
cp ./config.json "$TARGET_DIR/"