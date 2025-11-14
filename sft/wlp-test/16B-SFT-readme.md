
# 16B SFT 说明文档



## 数据集packing处理后的SFT

### 数据处理-packing

#### 数据处理脚本
```shell
#input_data_path=$1                # 设置输入文件路径
#tokenizer=$2                      # 设置分词器
#seq_len=$3                        # 设置训练用的序列长度
#output_data_path=$4               # 设置输出文件路径
#load_dir=$5                       # 设置HF模型的路径
#default_packing=$6                # 设置是否采用默认packing策略(默认false)

CURRENT_DIR="$(cd "$(dirname "$0")" && pwd)"
MEGATRON_PATH=$(dirname ${CURRENT_DIR})

export PYTHONPATH=${PYTHONPATH}:${MEGATRON_PATH}:${MEGATRON_PATH}/Megatron-LM_kuae_1022:${MEGATRON_PATH}/pai_megatron_patch

cd ../pai_megatron_patch/toolkits/sft_data_preprocessing

bash run_build_idxmap_sft_dataset_wlp.sh \
  /mnt/seed-program-nas/001688/datasets/SFT/tulu-3-sft-mixture/tulu_v3_mix.jsonl \
  HuggingFaceTokenizer \
  4096 \
  /mnt/seed-program-nas/001688/datasets/SFT/tulu-3-sft-mixture/tulu_v3_mix_zjllm_tokenizer_negeos_packing \
  /mnt/moer-train/public/models/zjllm-llama3-tokenizer \ 
  true
```
#### 处理后样本信息
packing前样本量: 93万条

packing后样本量: 

### 启动脚本中配置

```shell
...
if [ $SFT = true ]; then
  TRAINING_ARGS+=(
    --dataset MMAP
    --train-mode finetune
    --finetune
    
    # SFT不需要预训练模型checkpoint中的优化器状态以及随机数生产器状态
    --no-load-optim
    --no-load-rng
    
    # SFT一般需要训练利用数据集epoch次
    --dataloader-type cyclic
    
    # 该参数为了适配packing处理后的样本
    --reset-position-ids
  )
fi
...
```

#### 确定--train-iters

```bibtex
 train_iters = 数据集样本数 * epoches / global_batch_size
```













