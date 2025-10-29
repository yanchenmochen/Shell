import sys
import os 
sys.path.append('/mnt/self-define/songquanheng/Pai-Megatron-Patch/backends/megatron/PAI-Megatron-LM-240718')
import time
from megatron.core.datasets.indexed_dataset import IndexedDataset

# {os.environ['shome']}/dataset/mmap_tulu3_deepseekv2_sft_datasets-standard_text_document
# /mnt/self-define/songquanheng/dataset/mmap_tulu3_deepseekv2_sft_6000_zjlab
# {os.environ['shome']}/dataset/mmap_tulu3_deepseekv2_sft_6000_zjlab_text_document
# /mnt/self-define/songquanheng/dataset/mmap_tulu3_deepseekv2_sft_6000_zjlab
# 加载数据
dataset_pai = IndexedDataset(f"{os.environ['shome']}/dataset/mmap_tulu3_deepseekv2_sft_datasets-standard_text_document")

dataset_pai = IndexedDataset(f"{os.environ['seed_home']}/dataset/zjlab_tulu3_8192_full_packng_deepseek_v2_lite_sft_text_document")

dataset_zjlab = IndexedDataset(f"{os.environ['shome']}/dataset/mmap_tulu3_deepseekv2_sft_multi-turn_text_document")
dataset_zjlab = IndexedDataset(f"/mnt/self-define/songquanheng/dataset/zjlab_modify_tulu3_8192_Llama3-no-packing_sft_text_document")

# 访问前几个样本
for i in range(5):
    tokens = dataset_pai[i]   # numpy array of token IDs
    print(tokens[4096:4150])

# 找到 dataset_pai[10] 中小于 -100 的索引
tokens_10 = dataset_pai[10]  # 获取第10个样本
indices_less_than_neg100 = (tokens_10 < -100).nonzero()[0]  # 找到小于-100的索引
print(f"\ndataset_pai[10] 中小于 -100 的索引数量: {len(indices_less_than_neg100)}")
print(f"小于 -100 的索引: {indices_less_than_neg100}")
print(f"对应的值: {tokens_10[indices_less_than_neg100]}")
