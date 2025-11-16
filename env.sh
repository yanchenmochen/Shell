#!/bin/bash
export shome=/mnt/self-define/songquanheng
export seed_home="/mnt/seed-program-nas/001688/songquanheng"


alias dsv2="cd /mnt/self-define/songquanheng/Pai-Megatron-Patch/examples/deepseek_v2"
alias dsv2-zn="cd /mnt/self-define/zhangnan/Pai-Megatron-Patch/examples/deepseek_v2"
alias tulu3="cd /mnt/self-define/songquanheng/dataset/tulu3/data/tulu-3-sft-mixture/"
alias ds142ckpt="cd /mnt/self-define/zhangnan/checkpoints/DeepSeek-V2-Lite-to-mcore-tp1-pp4-ep2"
alias zn="cd /mnt/self-define/zhangnan"
alias sqh="cd /mnt/self-define/songquanheng"
alias pai-sft="cd /mnt/self-define/songquanheng/Pai-Megatron-Patch/toolkits/sft_data_preprocessing"
alias dsv2convert="cd /mnt/self-define/songquanheng/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/deepseek"
alias dsv2convert-zn="cd /mnt/self-define/zhangnan/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/deepseek"

alias zjlab-megatron="cd /mnt/self-define/songquanheng/zjlab-megatron"
alias pai-megatron="cd /mnt/self-define/songquanheng/Pai-Megatron-Patch"
export pai_megatron_home=/mnt/self-define/songquanheng/Pai-Megatron-Patch
export zjlab_megatron_home=/mnt/self-define/songquanheng/zjlab-megatron
alias pai-llama-convert="cd /mnt/self-define/songquanheng/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/llama"
alias pai-llama-train="cd /mnt/self-define/songquanheng/Pai-Megatron-Patch/examples/llama3_1"
alias pai-dsv2-train="cd /mnt/self-define/songquanheng/Pai-Megatron-Patch/examples/deepseek_v2"
alias pai-dsv3-train="cd /mnt/self-define/songquanheng/Pai-Megatron-Patch/examples/deepseek_v3"
alias pai-ds-convert="cd /mnt/self-define/songquanheng/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/deepseek"
alias zjlab-sft="cd /mnt/self-define/songquanheng/zjlab-megatron/toolkits/sft_data_preprocessing"
alias toolkit="cd /mnt/self-define/songquanheng/toolkits"

alias tools="cd /mnt/self-define/songquanheng/tools"


alias os="cd /mnt/self-define/dongjie/model/OpenCompass/opencompass"
alias zjlab-convert="cd /mnt/self-define/songquanheng/zjlab-megatron/toolkits/model_checkpoints_convertor"
alias zjlab-llama-sft="cd /mnt/self-define/songquanheng/zjlab-megatron/examples/optimal_case/A100/llama3.1/sft"
alias zjlab-llama-train="cd /mnt/self-define/songquanheng/zjlab-megatron/examples/training_scripts/llama3_1"
alias zjlab-llama-convert="cd /mnt/self-define/songquanheng/zjlab-megatron/toolkits/model_checkpoints_convertor/llama"
alias sft-llama="cd /mnt/self-define/songquanheng/output-Llama3_1-8b-sft/checkpoint/mcore-llama3-1-8B-sft"

export inference="/mnt/self-define/songquanheng/Pai-Megatron-Patch/backends/megatron/Megatron-LM-241113/examples/inference/llama_mistral"
alias 021="cd $seed_home/model/iter_0050000_hf_new"
alias 32b="cd $seed_home/Shell/huggingface/model/021-32B"
alias 16b="cd $seed_home/Shell/huggingface/model/021-16B"
alias 236b="cd $seed_home/Shell/huggingface/model/021-236B"
alias sft="cd $seed_home/Shell/sft"
alias cache="cd /root/.cache/huggingface/modules/transformers_modules/iter_0050000_hf_new"
export model_home="$([ -d /mnt/common/public/model ] && echo /mnt/common/public/model || echo /public/model)"
alias model="cd $model_home"
alias mcore="cd /mnt/seed-program-nas/001688/dongjie/X10000/zjlab-megatron/Megatron/Megatron-LM_old/examples/inference"
alias huggingface="cd $seed_home/Shell/huggingface"
alias log-dir="cd /mnt/seed-program-nas/001688/dongjie/X10000/zjlab-megatron/Megatron/Megatron-LM_old/examples/inference/output_mg_021"