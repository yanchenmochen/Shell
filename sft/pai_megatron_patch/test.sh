#! /bin/bash
set -ex
# 请在此处设置原始数据所在路径
data_dir=/mnt/dataset/wangluping/aimo/wlp_test_x10000

#开始数据清洗流程
dataset_dir=$(dirname $data_dir)
mkdir -p ${dataset_dir}/cleaned_wudao_dataset
cd ${dataset_dir}/cleaned_wudao_dataset
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama2-codes/preprocess_wudao2.py
# 此处与上一节不同，增加了key参数设为text
python preprocess_wudao2.py -i ${data_dir} -o ${dataset_dir}/cleaned_wudao_dataset -k text -p 32

# 合并清洗后的数据
mkdir ${dataset_dir}/wudao
cd ${dataset_dir}/wudao
find ${dataset_dir}/cleaned_wudao_dataset -name "*.json" -exec cat {} + >${dataset_dir}/wudao/merged_wudao_cleaned.json
rm -rf ${dataset_dir}/cleaned_wudao_dataset
