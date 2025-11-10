#!/bin/bash
# export MP_PP0_LAYERS=6 
# bash hf2mcore_deepseek_v2_moe_convertor.sh \
# A2.4B \
# /public/model/DeepSeek-V2-Lite \
# /mnt/self-define/dongjie/model/hf2mcore/DeepSeek-V2-Lite-to-mcore-tp1-pp4-ep8  \
# 1  \
# 4  \
# 8 \
# bf16 \
# false

### hf2mcore


### mcore2hf
# bash hf2mcore_deepseek_v2_moe_convertor_012.sh \
# 32B \
# /mnt/testlustre/test/dongjie/zjlab-megatron/zj_examples/deepseek_v2/logs_fp8vsbf16/tp1_pp1_ep8_dp256_mbs1_numbs32_gbs8192_2025-08-09_20:25_FP8false/checkpoint  \
# /mnt/testlustre/test/dongjie/zjlab-megatron/zj_examples/deepseek_v2/logs_fp8vsbf16/tp1_pp1_ep8_dp256_mbs1_numbs32_gbs8192_2025-08-09_20:25_FP8false/checkpoint-hf \
# 1  \
# 1  \
# 8 \
# bf16 \
# true \
# /mnt/testlustre/test/dongjie/zjlab-megatron/toolkits/model_checkpoints_convertor/deepseek/021-32b \
# /mnt/x10000/002266/models/zjllm-llama3-tokenizer

# 021-50000
#bash hf2mcore_deepseek_v2_moe_converter_021.sh \
#32B \
#/mnt/seed17/001688/checkpoint/021_32B \
#/mnt/seed-program-nas/001688/honghong/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/moe32b/iter_0025000_hf_1107 \
#1  \
#2  \
#8 \
#bf16 \
#true \
#/mnt/seed17/001688/checkpoint/021_32B/model_checkpoints_convertor/deepseek/021-32b \
#/mnt/seed17/001688/checkpoint/021_32B/zjllm-llama3-tokenizer

bash hf2mcore_deepseek_v2_moe_converter_021_30000.sh \
32B \
/mnt/seed-program-nas/001688/songquanheng/model/megatron014/ \
/mnt/seed-program-nas/001688/honghong/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/moe32b/iter_0030000_hf_1110 \
1  \
1  \
8 \
bf16 \
true \
/mnt/seed17/001688/checkpoint/021_32B/model_checkpoints_convertor/deepseek/021-32b \
/mnt/seed17/001688/checkpoint/021_32B/zjllm-llama3-tokenizer
