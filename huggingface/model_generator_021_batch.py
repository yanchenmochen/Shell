import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

import torch_musa

# 设置模型路径
# model_path = "/mnt/self-define/songquanheng/output-Llama3_1-8b-sft/checkpoint/mcore-llama3-1-8B-sft-iter3000"
# /mnt/hw-nas/002147/yanjun/data/pretrain/checkpoint/12.8T/tp1_pp2_ep8_mbs2_gbs4800-iter50000-hf 为021-32B转换出来的检查点
model_path = "/mnt/seed17/001688/honghong/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/moe32b/iter_0050000_hf_new"
device = torch.device("musa:4" if torch.musa.is_available() else "cpu")
# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,  # 使用 bfloat16 精度，确保硬件支持
).to(device)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
prompts = [
    "who are you?",
    "what is the result of 1+1?"
    "写一首关于夏天海边风景的短诗。",
    "小王有5个苹果，给了小李2个，还剩多少？",
    "请将下面的英文翻译成中文：‘Deep learning models require large amounts of data.’",
    "从前有一位勇敢的探险家，他踏上了一段未知的旅程，然后……",
    "Explain the difference between supervised and unsupervised learning.",
    "Write a short poem about a rainy afternoon.",
    "If John has 10 oranges and gives 3 to Mary, how many does he have left?",
    "Translate the following sentence into Chinese: 'Artificial intelligence is transforming industries.'",
    "Once upon a time, there was a mysterious island where strange creatures lived. Continue the story..."
]
texts = [prompt, text]
texts += prompts
model_inputs = tokenizer(texts, padding=True, return_tensors="pt").to(model.device)


# 确保 pad_token 有效
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 获取模型默认的生成配置
generation_config = GenerationConfig.from_pretrained(model_path)
generation_config.pad_token_id = tokenizer.pad_token_id  # 确保 pad_token_id 正确设置


# 5. 使用模型进行批量生成
with torch.no_grad():  # 禁用梯度计算以节省内存
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,  # 控制新生成token的最大数量
        num_return_sequences=1,  # 每个输入生成一个序列
        temperature=0.7,  # 控制生成随机性（可选）
        do_sample=True,  # 是否使用采样（可选）
        pad_token_id=tokenizer.pad_token_id  # 明确指定填充token的ID
    )


print("=== 批量生成结果 ===")
for i, (input_text, output_ids) in enumerate(zip(texts, generated_ids)):
    # 计算输入文本的长度（用于从生成结果中剔除输入部分）
    input_length = model_inputs['input_ids'][i].shape[0]
    
    # 提取新生成的部分（从输入结束位置开始）
    generated_tokens = output_ids[input_length:]
    
    # 将token解码为文本
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # 打印每个prompt和对应的生成内容
    print(f"\n--- 示例 {i+1} ---")
    print(f"输入(Prompt): {input_text}")
    print(f"生成内容: {generated_text}")