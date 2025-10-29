from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch import nn

def add_special_tokens_and_init(model, tokenizer, special_tokens):
    """
    自动添加特殊 token，并根据 embedding/lm_head 的大小自动扩展或初始化。
    """
    old_vocab_size = len(tokenizer)
    # 1. 添加特殊 token
    num_added_toks = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    if num_added_toks == 0:
        print("没有新增 token")
        return model, tokenizer

    print(f"新增 {num_added_toks} 个 token: {special_tokens}")

    # 2. 保存旧 embedding / lm_head
    old_embed = model.model.get_input_embeddings().weight.data.clone()
    old_lm_head = model.lm_head.weight.data.clone()
    
    old_embedding_size, hidden_size = old_embed.shape
    new_vocab_size = len(tokenizer)
    in_features = model.lm_head.in_features
    lm_head_out_features = model.lm_head.out_features

    # 3. 检查 embedding 大小是否够
    if old_embedding_size < new_vocab_size:
        print(f"扩展 embedding 从 {old_embedding_size} → {new_vocab_size}")
        model.resize_token_embeddings(new_vocab_size)
        new_embed = model.get_input_embeddings().weight.data
    else:
        print("embedding 大小足够，无需扩展")
        new_embed = model.get_input_embeddings().weight.data

    # 4. 计算均值
    mean_embed = old_embed.mean(dim=0)
    mean_lm_head = old_lm_head.mean(dim=0)

    # 5. 找出新增 token 的索引
    added_token_indices = torch.arange(old_vocab_size, new_vocab_size)

    # 6. 初始化新增 embedding
    with torch.no_grad():
        for idx in added_token_indices:
            new_embed[idx] = mean_embed

    # 7. 检查 lm_head 是否够大
    if lm_head_out_features < new_vocab_size:
        print(f"扩展 lm_head 从 {lm_head_out_features} → {new_vocab_size}")
        # 创建新的 lm_head
        new_lm_head_layer = nn.Linear(in_features, new_vocab_size, bias=False)
        with torch.no_grad():
            # 拷贝旧权重
            new_lm_head_layer.weight[:lm_head_out_features] = old_lm_head
            # 初始化新增 token 权重
            for idx in added_token_indices:
                new_lm_head_layer.weight[idx] = mean_lm_head
        model.lm_head = new_lm_head_layer
    else:
        print("lm_head 大小足够，无需扩展")
        with torch.no_grad():
            for idx in added_token_indices:
                model.lm_head.weight[idx] = mean_lm_head

    print("新增 token 初始化完成")
    return model, tokenizer

if __name__ == "__main__":
    
    model_name = "/public/model/DeepSeek-V2-Lite"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    special_tokens = ["<think>", "</think>"]

    model, tokenizer = add_special_tokens_and_init(model, tokenizer, special_tokens)

    # 保存
    save_dir = "/mnt/self-define/songquanheng/model/DeepSeek-V2-Lite-with-think-token"
    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(save_dir)
    print(f"模型和 tokenizer 已保存到 {save_dir}")
