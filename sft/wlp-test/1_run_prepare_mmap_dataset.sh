#input_data_path=$1                # 设置输入文件路径
#tokenizer=$2                      # 设置分词器
#seq_len=$3                        # 设置训练用的序列长度
#output_data_path=$4               # 设置输出文件路径
#load_dir=$5                       # 设置HF模型的路径
#default_packing=$6                # 设置是否采用默认packing策略(默认false)

CURRENT_DIR="$(cd "$(dirname "$0")" && pwd)"
MEGATRON_PATH=$(dirname ${CURRENT_DIR})

export PYTHONPATH=${PYTHONPATH}:${MEGATRON_PATH}:${MEGATRON_PATH}/Megatron-LM_kuae_1022:${MEGATRON_PATH}/pai_megatron_patch

#cd ../pai_megatron_patch/toolkits/sft_data_preprocessing
#export DEBUG=true
#bash run_build_idxmap_sft_dataset_wlp.sh \
#  /mnt/seed17/001688/wangluping/test-data/qwen_sft.json \
#  HuggingFaceTokenizer \
#  4096 \
#  /mnt/seed17/001688/wangluping/test-data/qwen_sft_negeos \
#  /mnt/moer-train/public/models/zjllm-llama3-tokenizer

cd ../pai_megatron_patch/toolkits/sft_data_preprocessing

bash run_build_idxmap_sft_dataset_wlp.sh \
  /mnt/seed-program-nas/001688/datasets/SFT/tulu-3-sft-mixture/tulu_v3_mix.jsonl \
  HuggingFaceTokenizer \
  4096 \
  /mnt/seed-program-nas/001688/datasets/SFT/tulu-3-sft-mixture/tulu_v3_mix_zjllm_tokenizer_negeos \
  /mnt/moer-train/public/models/zjllm-llama3-tokenizer
