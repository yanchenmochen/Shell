#input_data_path=$1                # 设置输入文件路径
#tokenizer=$2                      # 设置分词器
#seq_len=$3                        # 设置训练用的序列长度
#output_data_path=$4               # 设置输出文件路径
#load_dir=$5                       # 设置HF模型的路径
#default_packing=$6                # 设置是否采用默认packing策略(默认false)


cd /mnt/seed17/001688/wangluping/zjlab-megatron/pai_megatron_patch/toolkits/sft_data_preprocessing

bash run_build_idxmap_sft_dataset_wlp.sh \
/mnt/self-define/dongjie/model/dataset/tulu3/data/tulu-3-sft-mixture/tulu_v3_mix.jsonl \
HuggingFaceTokenizer \
4096 \
/mnt/self-define/dongjie/model/dataset/tulu3/data/tulu-3-sft-mixture/tulu_v3_mix \
/mnt/moer-train/public/models/zjllm-llama3-tokenizer

#/mnt/self-define/dongjie/model/dataset/tulu3/data/tulu-3-sft-mixture/tulu_v3_mix.jsonl
#/mnt/seed17/001688/wangluping/test-data/qwen_sft.json \

