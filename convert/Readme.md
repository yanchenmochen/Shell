# <center>脚本使用指南</center>

# 简介

convert是一款模型ckpt转换工具，方便用户低门槛的将megatron格式的ckpt转换到huggingface格式。

# 目录简介

```
run_021.sh
hf2mcore_deepseek_v2_moe_converter_021.sh

run_021.sh为启动脚本
传入参数定义如下，使用时根据实际情况修改
MODEL_SIZE=$1
SOURCE_CKPT_PATH=$2
TARGET_CKPT_PATH=$3
TP=$4
PP=$5
EP=$6
PR=$7
MG2HF=$8
HF_CKPT_PATH=$9
TOKENIZED_MODEL=$10

hf2mcore_deepseek_v2_moe_converter_021.sh为执行脚本
使用前根据实际情况对PYTHONPATH、LD_LIBRARY_PATH和模型参数进行配置。

```


# 使用指南

在shell脚本组使用过程中，使用的步骤如下：

## 配置脚本

根据实际情况配置run_021.sh和hf2mcore_deepseek_v2_moe_converter_021.sh文件。

## 执行

bash run_021.sh

