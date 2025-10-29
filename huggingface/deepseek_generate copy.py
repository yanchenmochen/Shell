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
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).musa()
model.generation_config = GenerationConfig.from_pretrained(model_path)
model.generation_config.pad_token_id = model.generation_config.eos_token_id

# # texts = ["中国的的首都是", "Pytorch is a very powerful programming tool, it can ", "I am a chinese man, I like to eat"]
# messages = [{"role": 'user', 'content': "我有两个箱子，其中每个箱子有3个苹果，我给了小明2个苹果，那么我还有多少个苹果？"}]
# # text = "我有两个箱子，其中每个箱子有3个苹果，我给了小明2个苹果，那么我还有多少个苹果？"
# res = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# print(res+'<think>')
# text = res+'<think>'
# inputs = tokenizer(text, return_tensors="pt",padding=True)
# outputs = model.generate(**inputs.to(model.device), max_new_tokens=1024)
# print(outputs)

# result = tokenizer.decode(outputs[0], skip_special_tokens=False)
# print(result)


# 使用加载的模型和分词器创建生成文本的pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device="musa")

# # 生成文本
# output = generator("Pytorch is a very powerful programming tool, it can ", max_length=50, num_return_sequences=1)
# print(output)
# output = generator("I am a chinese man, I like to eat", max_length=50, num_return_sequences=1, truncation=True, clean_up_tokenization_spaces = False)
# print(output)

prompts = [
    # 中文 prompts
    "请解释一下人工智能和机器学习的区别。",
    "写一首关于夏天海边风景的短诗。",
    "小王有5个苹果，给了小李2个，还剩多少？",
    "请将下面的英文翻译成中文：‘Deep learning models require large amounts of data.’",
    "从前有一位勇敢的探险家，他踏上了一段未知的旅程，然后……",

    # English prompts
    "Explain the difference between supervised and unsupervised learning.",
    "Write a short poem about a rainy afternoon.",
    "If John has 10 oranges and gives 3 to Mary, how many does he have left?",
    "Translate the following sentence into Chinese: 'Artificial intelligence is transforming industries.'",
    "Once upon a time, there was a mysterious island where strange creatures lived. Continue the story..."
]


out_batch = generator(prompts,  # 直接传列表
                      max_length=50,
                      num_return_sequences=1,
                      padding=True,  # 确保填充
                      truncation=True,
                      clean_up_tokenization_spaces=False)

for i, o in enumerate(out_batch):
    print(f"\nPrompt {i}: {o}")