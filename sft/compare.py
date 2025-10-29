import torch
import os
import numpy as np
from typing import Union, List
try:
    import torch_musa  # noqa: F401
except Exception:
    pass
# /mnt/self-define/dongjie/model/Pai-Megatron-Patch/Megatron-LM-241113/examples/inference/gpt/output_hidden_states
# /mnt/self-define/dongjie/model/Pai-Megatron-Patch-0923/Pai-Megatron-Patch/backends/megatron/Megatron-LM-241113/examples/inference/output_mg_bak
# /mnt/seed-program-nas/001688/dongjie/X10000/zjlab-megatron/Megatron/Megatron-LM_old/examples/inference/output_mg_021 02132b
os.chdir('/mnt/seed-program-nas/001688/dongjie/X10000/zjlab-megatron/Megatron/Megatron-LM_old/examples/inference/output_mg_021')
device = torch.device('muda:0' if torch.cuda.is_available() else 'cpu')


def calculate_mae(hf_tensor, mg_tensor, bins=10):
    # 计算两个张量之间的MAE和最大绝对值差
    absolute_diff = torch.abs(hf_tensor.flatten() - mg_tensor.flatten())
    mae = torch.mean(absolute_diff).item()
    max_diff = torch.max(absolute_diff).item()

    min_val = torch.min(absolute_diff).item()
    max_val = max_diff  # 就是上面计算的max_diff
    # 计算每个桶的统计数量
    bin_counts = torch.histc(absolute_diff.type(torch.float32), bins=bins, min=min_val, max=max_val)

    # 计算每个桶的边界值（用于理解每个桶代表的数值范围）
    bin_edges = torch.linspace(min_val, max_val, bins + 1)

    # 打印统计结果
    print(f"差值范围: [{min_val:.6f}, {max_val:.6f}]")
    print("-" * 50)
    
    for i in range(bins):
        lower_edge = bin_edges[i].item()
        upper_edge = bin_edges[i+1].item()
        count = bin_counts[i].item()
        print(f"[{lower_edge:.6f}, {upper_edge:.6f})\t{int(count)} \t {int(count)*100/absolute_diff.numel():.2f}%")

    # 打印基本的统计信息
    # print("-" * 50)
    # print(f"平均值 (MAE): {mae:.6f}")
    # print(f"最大值: {max_diff:.6f}")
    # print(f"中位数: {torch.median(absolute_diff).item():.6f}")
    # print(f"标准差: {torch.std(absolute_diff).item():.6f}")
    print(f"总样本数: {absolute_diff.numel()}")
    return mae, max_diff


def compare_layer_output(layer: int, name: Union[str, List[str]]) -> None:
    """
    比较指定层的 HF 与 MG 输出张量差异并打印 MAE
    
    参数:
        layer (int): 层号
        name (Union[str, List[str]]): 输出类型，可以是字符串或字符串列表，
                                     如 'mlp_output'、'attn_output' 或 ['input', 'attn_input', 'mlp_input']
        device (str): 设备，如 'cuda' 或 'cpu'
    """
    # 统一处理参数：将字符串转换为列表
    names = [name] if isinstance(name, str) else name
    
    for name in names:
        hf_path = f"hf_layer_{layer}_{name}.pt"
        mg_path = f"mg_layer_{layer}_{name}.pt"
        
        hf_output = torch.load(hf_path, map_location=device)
        mg_output = torch.load(mg_path, map_location=device)
        
        print(f"\n===== Layer {layer} {name} Comparison =====")
        print(f"HF {name} Tensor: shape: {hf_output.shape} \n{hf_output} ")
        print(f"MG {name} Tensor: shape: {mg_output.shape} \n{mg_output} ")
        print(f"layer{layer} {name} diff ", calculate_mae(hf_output.to(device), mg_output.to(device)))
    

'''
hf_layer_0_attn_input.pt
hf_layer_0_attn_output.pt
hf_layer_0_input.pt
hf_layer_0_mlp_input.pt
hf_layer_0_mlp_output.pt


hf_layer_16_attn_input.pt
hf_layer_16_attn_output.pt
hf_layer_16_expert.output.pt
hf_layer_16_expert_input.pt
hf_layer_16_expert_output.pt
hf_layer_16_gate_input.pt
hf_layer_16_gate_output.pt
hf_layer_16_input.pt
hf_layer_16_mlp_input.pt
hf_layer_16_mlp_output.pt
hf_layer_16_moe_input.pt
hf_layer_16_moe_output.pt
hf_layer_16_share_expert.outupt.pt
hf_layer_16_share_expert_input.pt
'''
for layer in range(20):
    base_outputs = ["input", "attn_input", "attn_output", "attn_residual_output", "mlp_input", "mlp_output", "mlp_residual_output"]
     
    if layer > 0:
        moe_outputs = ['moe_input', 'gate_input', 'expert_input', 'expert_output','share_expert_input', 'share_expert_output', 'moe_output']
        base_outputs = base_outputs[:-1] + moe_outputs + base_outputs[-1:]    
    compare_layer_output(layer=layer, name=base_outputs)
