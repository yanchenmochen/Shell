import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# 设置模型路径
model_path = "/mnt/self-define/songquanheng/model/DeepSeek-V2-Lite-with-think-token"
model_path = "/mnt/seed-program-nas/001688/songquanheng/model/iter_0050000_hf_new"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




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
generation_config.use_cache = False  # 禁用生成阶段缓存，确保每步不复用KV

# 同步关闭模型级缓存，确保 forward 不写入/读取 past_key_values
model.config.use_cache = False
# 可选：将本地 generation_config 绑定到模型，避免 generate 内部拉取预训练默认配置
model.generation_config = generation_config

# 定义 prompts 列表
# prompts = [
#     "请解释一下人工智能和机器学习的区别。",
#     "写一首关于夏天海边风景的短诗。",
#     "小王有5个苹果，给了小李2个，还剩多少？",
#     "请将下面的英文翻译成中文：‘Deep learning models require large amounts of data.’",
#     "从前有一位勇敢的探险家，他踏上了一段未知的旅程，然后……",
#     "Explain the difference between supervised and unsupervised learning.",
#     "Write a short poem about a rainy afternoon.",
#     "If John has 10 oranges and gives 3 to Mary, how many does he have left?",
#     "Translate the following sentence into Chinese: 'Artificial intelligence is transforming industries.'",
#     "Once upon a time, there was a mysterious island where strange creatures lived. Continue the story..."
# ]

prompts=[
    'who are you?'
]

import os
if os.getenv('DEBUG', '0').lower() in ('1', 'true', 'yes'):
    import debugpy
    try:
        # 使用异常处理适配多进程代码，这样只有一个进程会监听5678端口
        debugpy.listen(("localhost", 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        print("Debugger attached")
    except Exception as e:
        # 如果端口已被占用，忽略异常（可能是其他进程已启动调试）
        print(f"调试器启动失败: {e}")
        # 或者更详细的信息
        print(f"调试器启动失败，类型: {type(e).__name__}, 详情: {e}")
        pass

# 逐个处理每个 prompt
for i, prompt in enumerate(prompts):
    # 编码输入文本
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,    # 启用填充以确保批处理兼容性
        truncation=True, # 启用截断以防止输入过长
        max_length=512   # 设置最大输入长度
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
            do_sample=True,    # 启用采样以产生更多样化的输出
            temperature=0.7,   # 控制采样随机性
            pad_token_id=tokenizer.pad_token_id,  # 明确指定 pad_token_id
            use_cache=False
        )
    
    # 解码生成结果
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nPrompt {i}: {generated_text}")