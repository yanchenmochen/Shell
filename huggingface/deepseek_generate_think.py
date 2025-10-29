import torch
import torch_musa
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, pipeline

# model_name = "/mnt/seed-program-nas/001688/zn/output_tulu3/checkpoint/finetune-mcore-deepseek-v2-A2.4B-lr-5e-6-minlr-1e-6-bs-2-gbs-1024-seqlen-4096-pr-bf16-tp-1-pp-4-cp-1-ac-sel-do-true-sp-true-ti-10000-wi-100/iter_0000900-hf"
model_dir= "/mnt/seed-program-nas/001688/zn/output_tulu3/checkpoint/finetune-mcore-deepseek-v2-A2.4B-lr-5e-6-minlr-1e-6-bs-2-gbs-1024-seqlen-4096-pr-bf16-tp-1-pp-4-cp-1-ac-sel-do-true-sp-true-ti-10000-wi-100/"
model_name = 'Deepseek-V2-Lite-tp1-pp4-ep2-iter2700'
model_path=model_dir+model_name
model_path="/mnt/self-define/songquanheng/model/DeepSeek-V2-Lite-with-think-token"
model_path="/mnt/seed-program-nas/001688/sft_ckpt/20250921-125550_finetune-mcore-deepseek-v2-A2.4B-lr-5e-6-minlr-1e-6-bs-1-gbs-256-seqlen-8192-pr-bf16-tp-1-pp-1-cp-1-ac-full-do-true-sp-true-ti-8000-wi-200/iter_0000500-hf"
# model_path="/mnt/self-define/songquanheng/model/DeepSeek-V2-Lite-with-think-token-megatron-Deepseek-back-tp1pp4ep2iter" 
# model_path="/public/model/DeepSeek-V2-Lite"
model_path="/mnt/seed-program-nas/001688/songquanheng/output/cot-0923-1514/checkpoint/finetune-mcore-deepseek-v2-A2.4B-lr-5e-6-minlr-1e-6-bs-1-gbs-128-seqlen-8192-pr-bf16-tp-1-pp-4-cp-1-ac-sel-do-true-sp-true-ti-10000-wi-100/Deepseek-V2-Lite-tp1-pp4-ep2-iter7500"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).musa()
model.generation_config = GenerationConfig.from_pretrained(model_path)
model.generation_config.pad_token_id = model.generation_config.eos_token_id

prompt="<think> and </think> is special tokens"
input_ids = tokenizer.encode(prompt, add_special_tokens=False)
print(tokenizer.encode("<think> and </think> is special tokens"))
print(tokenizer.encode("<think> </think>", add_special_tokens=False))
# texts = ["中国的的首都是", "Pytorch is a very powerful programming tool, it can ", "I am a chinese man, I like to eat"]
messages = [{"role": 'user', 'content': "我有两个箱子，其中每个箱子有3个苹果，我给了小明2个苹果，那么我还有多少个苹果？"}, {'role': 'assistant', "content": '<think>\n'}]
# text = "我有两个箱子，其中每个箱子有3个苹果，我给了小明2个苹果，那么我还有多少个苹果？"
res = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
# print(res+'<think>')
text = res
inputs = tokenizer(text, return_tensors="pt",padding=True, add_special_tokens=False)
print(f"inputs: {inputs}")
outputs = model.generate(**inputs.to(model.device), max_new_tokens=1024)
print(f"outputs: {outputs}")

result = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(result)

inputs = tokenizer("我有两个箱子，其中每个箱子有3个苹果，我给了小明2个苹果，那么我还有多少个苹果？", return_tensors="pt")
outputs = model.generate(**inputs.to(model.device), max_new_tokens=1024)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

texts = [
    "Hello world!",
    "Hugging Face makes NLP easy.",
    "我喜欢学习大模型。",
    "写一首春天的诗歌，请对仗一些",
    "中国的首都是"
]

# 批量编码
encodings = tokenizer(texts,               # 直接传list
                      padding=True,        # 按最大长度补齐
                      truncation=True,     # 超过max_length截断
                      max_length=20,
                      return_tensors="pt") # 返回 PyTorch tensor

outputs = model.generate(
    **encodings.to(model.device),
    max_new_tokens=32,
    do_sample=False  # 不随机采样，方便看结果
)

print(f"outputs: {outputs}")

decoded_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(f"decoded_texts: {decoded_texts}")
for i, t in enumerate(decoded_texts):
    print(f"输入：{texts[i]} → 生成：{t}")