---
aliases: []
tags: []
description:
framework:
date created: 2025-10-30 星期四 14:39
date modified: 2025-11-03 星期一 14:46
---

***
# Deepseek_v2_lite base/sft 测评对齐

## 环境版本

1：transformer4.55.0  
2：vllm 0.10.1.1  
3：OpenCompass 0.5.0  

## 测评配置
```python
models = [
    dict(
        type=VLLM, #VLLMwithChatTemplate,HuggingFaceBaseModel
        abbr='1018-iter8000-vllm-tulu3-a100',
        path=TARGET_PATH,
        new-source-iter50000-hf',
	        model_kwargs=dict(
            tensor_parallel_size=2,
            #torch_dtype='torch.bfloat16',
            dtype='auto',
            ), #gpu_memory_utilization=0.8
        max_seq_len=4096,
        max_out_len=512,#512
        batch_size=16,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=2, num_procs=1),
    )
]
```

| 模型             | 测试数据      | 数据版本                            | type | base 测评分数 | base分数 | 误差    | sft 测评分数 | sft base 分数 | 误差    |
| -------------- | --------- | ------------------------------- | ---- | --------- | ------ | ----- | -------- | ----------- | ----- |
| deepseekV2Lite | mmlu      | mmlu_ppl_ac766d                 | VLLM | 58.31     | 58.3   | 0.01  | 53.41    | 55.7        | -2.29 |
| deepseekV2Lite | gsm8k     | gsm8k_gen_17d0dc                | VLLM | 40.86     | 41.1   | -0.24 | 69.52    | 72          | -2.48 |
| deepseekV2Lite | humaneval | deprecated_humaneval_gen_a82cae | VLLM | 28.05     | 29.9   | -1.85 | 56.1     | 57.3        | -1.2  |
| deepseekV2Lite | math      | math_4shot_base_gen_db136b      | VLLM | 15.44     | 17.1   | -1.66 | 25.72    | 27.9        | -2.18 |

    其中humaneval需要更换老版本的测评函数，否则测评分数一直为0
具体过程为：在这里加上原来的后处理方法
![1](https://gitee.com/songquanheng/imgs/raw/master/2025//202510291734970.png)  
然后再config/datasets里import这个方法，就可以切换成原来的后处理方法  
![2](https://gitee.com/songquanheng/imgs/raw/master/2025//202510291734415.png)

# Deepseek_v2_lite sft 测评对齐

    根据deepseek_v2_lite base 测评对齐的结果，在deepseek_v2_lite sft 测评对齐中模型type全部使用VLLM格式，测评环境版本与模型配置同deepseek_v2_lite_base测评  

## type = VLLM 时测评结果
| 测试数据 | 数据版本 | base | 0 | 2000 | 4000 | 6000 | 8000 | 10000 | 12000 | 14000 | 16000 | 18000 |20000 | sft base |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mmlu | mmlu_ppl_ac766d | 58.3 | 58.31 | 58.70 | 58.03 | 57.71 | 58.75 | 58.60 | 58.73 | 59.16 | 59.38 | 59.42 | 59.53 | 55.7 |
| gsm8k | gsm8k_gen_17d0dc | 41.1 | 40.86 | 44.88 | 49.36 | 53.30 | 55.95 | 55.95 | 55.80 | 56.63 | 56.79 | 56.25 | 55.88 | 72 |
| humaneval | deprecated_humaneval_gen_a82cae | 29.9 | 28.05 | 29.88 | 32.32 | 32.32 | 29.88 | 32.93 | 32.93 | 32.32 | 34.15 | 31.71 | 32.93 | 57.3 |
| math | math_4shot_base_gen_db136b | 17.1 | 15.44 | 18.30 | 19.92 | 20.62 | 20.16 | 21.00 | 21.90 | 21.32 | 20.62 | 21.34 | 21.86 | 27.9 |
![33dd562247945f9d15af3c74d992996f.png](https://gitee.com/songquanheng/imgs/raw/master/2025//202510300927326.png)

## type = VLLMwithChatTemplate 时测评结果
| 测试数据 | 数据版本 | base | 0 | 2000 | 4000 | 6000 | 8000 | 10000 | 12000 | 14000 | 16000 | 18000 |20000 | sft base |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mmlu | mmlu_ppl_ac766d | 58.3 | 58.31 | 58.13 | 57.81 | 57.5 | 58.36 | 58.66 | 58.47 | 58.85 | 59.12 | 56.63 | 59.34 | 55.7 |
| gsm8k | gsm8k_gen_17d0dc | 41.1 | 40.86 | 46.93 | 51.25 | 52.31 | 54.28 | 57.01 | 57.54 | 56.94 | 57.39 | 56.25 | 58.53 | 72 |
| humaneval | deprecated_humaneval_gen_a82cae | 29.9 | 18.29 | 34.15 | 33.54 | 37.8 | 42.68 | 43.29 | 40.85 | 42.68 | 42.68 | 43.9 | 42.07 | 57.3 |
| math | math_4shot_base_gen_db136b | 17.1 | 3.74 | 18.64 | 19.04 | 20.28 | 20.88 | 21.64 | 21.96 | 20.1 | 20.56 | 21.72 | 22.48 | 27.9 |
![9a50c456601ff6069ecee10c417872c8.png](https://gitee.com/songquanheng/imgs/raw/master/2025//202510300926708.png)

## 综合测评结果
    在对sft流程进行综合测评时，初始模型应该使用VLLM模型type + ppl数据集进行测评，进入sft过程后应该使用VLLM with chat template模型type + gen数据集进行测评。但gsm8k/humaneval/math这三个数据集opencompass只提供了gen模型，综合考虑后，base模型使用VLLM type,mmlu 使用ppl数据，gsm8k/humaneval/math使用gen数据。进入sft过程后所有check point使用 VLLM with chat template type 和 gen数据进行测评。

| 测试数据      | 数据版本                            | base | 0     | 2000  | 4000  | 6000  | 8000  | 10000 | 12000 | 14000 | 16000 | 18000 | 20000 | sft base |
| --------- | ------------------------------- | ---- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | -------- |
| mmlu      | mmlu_ppl_ac766d/mmlu_gen_4d595a | 58.3 | 58.31 | 58.23 | 58.10 | 57.05 | 58.10 | 58.08 | 58.40 | 59.05 | 58.89 | 59.38 | 59    | 55.7     |
| gsm8k     | gsm8k_gen_17d0dc                | 41.1 | 40.86 | 46.93 | 51.25 | 52.31 | 54.28 | 57.01 | 57.54 | 56.94 | 57.39 | 56.25 | 58.53 | 72       |
| humaneval | deprecated_humaneval_gen_a82cae | 29.9 | 28.05 | 34.15 | 33.54 | 37.8  | 42.68 | 43.29 | 40.85 | 42.68 | 42.68 | 43.9  | 42.07 | 57.3     |
| math      | math_4shot_base_gen_db136b      | 17.1 | 15.44 | 18.64 | 19.04 | 20.28 | 20.88 | 21.64 | 21.96 | 20.1  | 20.56 | 21.72 | 22.48 | 27.9     |
![1dfa0cec5514dd3e806e16535d203507.png](https://gitee.com/songquanheng/imgs/raw/master/2025//202511031302173.png)

***

# w64 hf-pretrain-5w 测评对齐

## 测评环境与配置

1：transformer4.55.0 -> 4.38.2 (Transformers 库版本与模型（Deepseek）代码中使用的缓存类接口不兼容，具体是DynamicCache对象没有get_usable_length方法，导致模型前向传播时调用失败)  
2：type = VLLM -> HuggingFaceBaseModel(与w64对齐)
## 测评分数 
| 测评环境 | 数据版本 | 测评分数 |
| --- | --- | --- |
| w64 | mmlu_ppl_ac766d | 48.23 |
| our | mmlu_ppl_ac766d | 48.23 |

各领域分数数据较多不便展示，但所有分数都是相同的  

# 测评文件实例

## docker

镜像：
```bash
10.200.88.53/framework/pai_megatron:torch24.07-py3-v8.1-opencompass050  
```

启动命令： 
```bash
docker run --name songquanheng-opencompass --env-file 
/mnt/nas_v1/common/public/config/docker.env --gpus all -v 
/mnt/nas_v1/common/public:/public -v /mnt:/mnt -v /mnt/self
define:/mnt/self-define --shm-size=128gb  --privileged --cap-add=ALL --pid=host --ipc=host --network=host  -it 
10.200.88.53/framework/pai_megatron:torch24.07-py3-v8.  
```

在使用的时候，添加 --network host 使得容器与宿主机共享网络，这样可以方便的通过
vscode 连接容器。 连接容器的时候，需要修改默认端口 
/etc/ssh/sshd_config   

实例容器：
```bash
7b6240f6687e   10.200.88.53/framework/pai_megatron:torch24.07-py3-v8.1-opencompass050                  "/opt/nvidia/nvidia_…"                                                                       dongjie-pai-opencompass050
```

## 组件版本

OpenCompass 0.5.0  
transformer 4.55.0   
vllm 0.10.1.1  

## config文件
```python
from mmengine.config import read_base
from opencompass.models import VLLM, VLLMwithChatTemplate, HuggingFaceCausalLM,HuggingFaceBaseModel

with read_base():
    
    from opencompass.configs.datasets.mmlu.mmlu_ppl_ac766d import mmlu_datasets
    from opencompass.configs.datasets.ceval.ceval_gen_5f30c7 import ceval_datasets 
    from opencompass.configs.datasets.hellaswag.hellaswag_10shot_gen_e42710 import hellaswag_datasets 
    from opencompass.configs.datasets.bbh.bbh_gen_98fba6 import bbh_datasets
    from opencompass.configs.datasets.aime2024.aime2024_gen_17d799 import aime2024_datasets  
    from opencompass.configs.datasets.livecodebench.livecodebench_gen_a4f90b import LCB_datasets  
    from opencompass.configs.datasets.humaneval.deprecated_humaneval_gen_a82cae import humaneval_datasets

    from opencompass.configs.datasets.mbpp.sanitized_mbpp_gen_742f0c import sanitized_mbpp_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_17d0dc import gsm8k_datasets
    from opencompass.configs.datasets.math.math_4shot_base_gen_db136b import math_datasets

    from opencompass.configs.datasets.agieval.agieval_mixed_0fa998 import agieval_datasets
    from opencompass.configs.datasets.cmmlu.cmmlu_0shot_cot_gen_305931 import cmmlu_datasets 

datasets = [*mmlu_datasets]
#tulu3 0924
SOURCE_PATH='/mnt/seed-program-nas/001688/sft_ckpt/musa-deepseek-v2-A2.4B-lr-2e-6-minlr-1e-6-bs-1-gbs-128-seqlen-4096-pr-bf16-tp-1-pp-1-cp-1-ac-sel-do-true-sp-false-ti-210000-wi-6300'

TARGET_PATH = f"{SOURCE_PATH}/iter_0025500-hf"
TARGET_PATH="/mnt/seed-program-nas/001688/sft_ckpt/sft14000-hf-1018"
models = [
    dict(
        type=VLLM, #VLLMwithChatTemplate,HuggingFaceBaseModel
        abbr='1018-iter8000-vllm-tulu3-a100',
        path=TARGET_PATH,
        model_kwargs=dict(
            tensor_parallel_size=2,
            #torch_dtype='torch.bfloat16',
            dtype='auto',
            ), #gpu_memory_utilization=0.8
        max_seq_len=4096,
        max_out_len=512,#512
        batch_size=4,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=2, num_procs=1),
    )
]
```
## 启动脚本
```bash
#!/bin/bash
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"  python run.py ./cdj_config_dsv2litesft_wzb.py  --max-num-workers 4 -w output/
```