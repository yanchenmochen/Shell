import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = torch.device("cuda" if torch.cuda.is_available()  else "cpu")

# model_dir="/public/model/DeepSeek-V2-Lite"
model_dir = "/mnt/self-define/songquanheng/model/DeepSeek-V2-Lite-with-think-token"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir,trust_remote_code=True, torch_dtype=torch.bfloat16).to(DEVICE)
model.eval()

print(f"原始词表长度: {len(tokenizer.vocab)}")
old_vocab_size = len(tokenizer.vocab) #100002
input="<think> with </think>"
print(f"{tokenizer.encode(input)}") # [100000, 27, 17249, 29, 366, 1119, 17249, 29] 表明无法识别<think> </think>标记

print(f"{tokenizer.special_tokens_map}") 
#{'bos_token': '<｜begin▁of▁sentence｜>', 'eos_token': '<｜end▁of▁sentence｜>', 'pad_token': '<｜end▁of▁sentence｜>', 'additional_special_tokens': ['<think>', '</think>']}

special_tokens_to_add = ['<think>', "</think>"]
tokenizer.add_special_tokens({'additional_special_tokens': special_tokens_to_add})
# {'bos_toke": '<｜begin▁of▁sentence｜>', 'eos_token': '<｜end▁of▁sentence｜>', 'pad_token': '<｜end▁of▁sentence｜>', 'additional_special_tokens': ['<think>', '</think>']}
print(f"{tokenizer.encode(input)}")
# [100000, 100002, 366, 207, 100003] 表明已经认识了两个特殊标记
print(f"扩展后词表长度: {len(tokenizer.vocab)}")
new_vocab_size = len(tokenizer)
print(f"{tokenizer.special_tokens_map}")
# {'bos_token': '<｜begin▁of▁sentence｜>', 'eos_token': '<｜end▁of▁sentence｜>', 'pad_token': '<｜end▁of▁sentence｜>', 'additional_special_tokens': ['<think>', '</think>']}

user_prompt = "请写一个 Python 函数，实现快速排序算法并注释每一步骤。"
prompt_with_think = f"{user_prompt} <think>"
inputs = tokenizer(prompt_with_think, return_tensors="pt").to(model.device)

max_new_tokens = 1024
temperature=0.7
top_p = 0.9
print(user_prompt)

output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id
    )

generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

result = prompt_with_think + generated + "</think>"
print(f"result: {result}")
print(f"之后len(tokenizer): {len(tokenizer)}")
# model.resize_token_embeddings(len(tokenizer))

