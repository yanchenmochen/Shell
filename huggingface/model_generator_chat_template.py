import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# 设置模型路径
# model_path = "/mnt/self-define/songquanheng/output-Llama3_1-8b-sft/checkpoint/mcore-llama3-1-8B-sft-iter3000"
model_path = "/public/model/DeepSeek-V2-Lite-Chat"
# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,  # 使用 bfloat16 精度，确保硬件支持
    device_map="auto"             # 自动分配设备
)

# 确保 pad_token 有效
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 获取模型默认的生成配置
generation_config = GenerationConfig.from_pretrained(model_path)
generation_config.pad_token_id = tokenizer.pad_token_id  # 确保 pad_token_id 正确设置

# 定义 prompts 列表
prompts = [
    "请解释一下人工智能和机器学习的区别。",
    # "写一首关于夏天海边风景的短诗。",
    # "小王有5个苹果，给了小李2个，还剩多少？",
    # "请将下面的英文翻译成中文：‘Deep learning models require large amounts of data.’",
    # "从前有一位勇敢的探险家，他踏上了一段未知的旅程，然后……",
    # "Explain the difference between supervised and unsupervised learning.",
    # "Write a short poem about a rainy afternoon.",
    # "If John has 10 oranges and gives 3 to Mary, how many does he have left?",
    # "Translate the following sentence into Chinese: 'Artificial intelligence is transforming industries.'",
    # "Once upon a time, there was a mysterious island where strange creatures lived. Continue the story..."
]

import os


import debugpy
try:#使用异常处理适配多进程代码，这样只有一个进程会监听5678端口
    debugpy.listen(("localhost", 5668))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    pass

# messages = [
#         [{"role": "user", "content": "Hello, who are you?"}],
#         # [{"role": "user", "content": "What is the result of 1+1?"}],
#         # [{"role": "user", "content": "\n\ndef truncate_number(number: float) -> float:\n    \"\"\" Given a positive floating point number, it can be decomposed into\n    and integer part (largest integer smaller than given number) and decimals\n    (leftover part always smaller than 1).\n\n    Return the decimal part of the number.\n    >>> truncate_number(3.5)\n    0.5\n    \"\"\"\n"}],
# ]   

messages = [
    [{"role": "user", "content": "Hello, who are you?"}],
    # [{"role": "user", "content": "What is the result of 1+1?"}],
    # [{"role": "user", "content": "Explain the difference between a list and a tuple in Python."}],
    # [{"role": "user", "content": "写一个 Python 函数，用来判断一个数字是否为质数。"}],
    # [{"role": "user", "content": "请解释 GPT 和 BERT 的主要区别。"}],
    # [{"role": "user", "content": "Translate 'Good morning' into French."}],
    # [{"role": "user", "content": "What is the time complexity of binary search?"}],
    # [{"role": "user", "content": "给出三个 Python 装饰器的示例。"}],
    # [{"role": "user", "content": "Explain what overfitting means in machine learning."}],
    # [{"role": "user", "content": "Summarize the story of 'The Little Prince' in three sentences."}],
]

for message in messages:
    # print(f"{message['role']}: {message['content']}\n")
    
    # add_generation_prompt=True 会在末尾添加 'Assistant:'，提示模型开始生成
    input_tensor = tokenizer.apply_chat_template(
        message,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt"
    )
    inputs = input_tensor.to(model.device)

    # 使用 model.generate() 进行文本生成
    with torch.no_grad():  # 禁用梯度计算以节省内存
        outputs = model.generate(
            inputs,
            max_new_tokens=512,  # 控制新生成 token 的数量
            num_return_sequences=1,
            generation_config=generation_config,  # 传入生成配置
            pad_token_id=tokenizer.pad_token_id  # 明确指定 pad_token_id
        )
    
    # 解码生成结果
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nPrompt {message}: {generated_text}")