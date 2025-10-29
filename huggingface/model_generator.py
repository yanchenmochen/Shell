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
generation_config.temperature = 0.6
generation_config.top_p = 0.95
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

# messages = [
#     {"role": "user", "content": "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\nFind the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.\n\nA) 0\nB) 4\nC) 2\nD) 6\n"},
# ]

# res = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# inputs = tokenizer(res, return_tensors="pt",padding=True, add_special_tokens=False)
# seed = 42
# torch.manual_seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(seed)

# # 确保CUDA操作的确定性（如果使用GPU）
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# 对于PyTorch 2.0+，设置以下环境变量
import os
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# 启用确定性算法
# torch.use_deterministic_algorithms(True)
# with torch.no_grad():  # 禁用梯度计算以节省内存
#     outputs = model.generate(
#         **inputs.to(model.device),
#         max_new_tokens=1024,  # 控制新生成 token 的数量
#         num_return_sequences=1,
#         generation_config=generation_config,  # 传入生成配置
#         do_sample=True,    # 启用采样以产生更多样化的输出

#         pad_token_id=tokenizer.pad_token_id  # 明确指定 pad_token_id
#     )
# generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(f"\n{generated_text}")

# 逐个处理每个 prompt

import debugpy
try:#使用异常处理适配多进程代码，这样只有一个进程会监听5678端口
    debugpy.listen(("localhost", 5668))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    pass
for i, prompt in enumerate(prompts):
    # 编码输入文本
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,    # 启用填充以确保批处理兼容性
        truncation=True, # 启用截断以防止输入过长
        max_length=1024   # 设置最大输入长度
    )
    
    # 将输入张量移动到模型所在的设备
    inputs = inputs.to(model.device)
    
    # 计算当前输入的序列长度，用于 cache_position
    input_length = inputs.input_ids.shape[1]



    # 使用 model.generate() 进行文本生成
    with torch.no_grad():  # 禁用梯度计算以节省内存
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,  # 控制新生成 token 的数量
            num_return_sequences=1,
            generation_config=generation_config,  # 传入生成配置
            do_sample=False,    # 启用采样以产生更多样化的输出
            temperature=1.0,   # 控制采样随机性
            top_k=0,
            top_p=1.0,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id  # 明确指定 pad_token_id
        )
    
    # 解码生成结果
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nPrompt {i}: {generated_text}")