import time,os
import torch
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
from utils import get_system_name, sync_device, get_torch_profiler

# 1. 构造测试输入数据
def generate_test_inputs(device, batch_size, seq_len,  num_q_heads, num_kv_heads, k_head_dim, v_head_dim, qkv_contiguous):
    # batch_size, seq_len, num_heads, head_dim = 4096, 1, 4, 128
    # 生成随机输入张量 (使用BF16格式以测试FlashAttention)
    if device == 'musa':
        if qkv_contiguous:
            query = torch.randn((batch_size, num_q_heads, seq_len,  k_head_dim), 
                        dtype=torch.bfloat16, device=device)
            key = torch.randn((batch_size, num_kv_heads, seq_len,  k_head_dim), 
                    dtype=torch.bfloat16, device=device)
            value = torch.randn((batch_size, num_kv_heads, seq_len, v_head_dim), 
                        dtype=torch.bfloat16, device=device)
        else:
            query = torch.randn((batch_size,  seq_len, num_q_heads,  k_head_dim), 
                        dtype=torch.bfloat16, device=device).transpose(1,2)
            key = torch.randn((batch_size,  seq_len,  num_kv_heads, k_head_dim), 
                    dtype=torch.bfloat16, device=device).transpose(1,2)
            value = torch.randn((batch_size, seq_len, num_kv_heads,   v_head_dim), 
                        dtype=torch.bfloat16, device=device).transpose(1,2)
    else:
        query = torch.randn((batch_size, seq_len, num_q_heads,  k_head_dim), 
                    dtype=torch.bfloat16, device=device)
        key = torch.randn((batch_size, seq_len, num_kv_heads, k_head_dim), 
                dtype=torch.bfloat16, device=device)
        value = torch.randn((batch_size, seq_len, num_kv_heads, v_head_dim), 
                    dtype=torch.bfloat16, device=device)
    
    return query, key, value

# 4. 基准测试函数
def benchmark_flashattention(q, k, v, warmup=10, repeat=100, device='musa'):
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    fwd_elapsed_time, bwd_elapsed_time = 0, 0
    profiler = get_torch_profiler(device, False)
    # Benchmark
    # sync_device(device)
    for i in range(repeat):
        if i == warmup:
            sync_device(device)
            fwd_start_time = time.time()
        # with torch.no_grad():
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
            # fwd
            # if device == 'musa':
            attn_output = torch.ops.aten._scaled_dot_product_attention_flash_musa(
                q, # batch_size, head_num , seq_len,  head_size
                k, # batch_size, head_num, seq_len, head_size
                v, #batch_size, head_num, seq_len, head_size
                dropout_p=0.0,
                is_causal=True,
            )
            print(f"=== FlashAttention输出: {attn_output}")
            # else:
            #     out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_forward(
            #         q = q,
            #         k = k,
            #         v = v,
            #         dropout_p = 0.0, 
            #         softmax_scale = 0.1,
            #         causal = True,
            #         window_size = (-1,0),
            #         alibi_slopes = None,
            #         return_softmax = False
            #     )
        profiler.step() if profiler else 1
        
    sync_device(device)
    fwd_elapsed_time = (time.time() - fwd_start_time) / (repeat-warmup) * 1000

    # print(f"=== FlashAttention输出: {attn_output}")
    # for i, o in enumerate(attn_output):
    #     print(f"=== 输出{i}: {type(o)}")
    # 测试反向传播

    sync_device(device)
    for i in range(repeat):
        if i == warmup:
            sync_device(device)
            bwd_start_time = time.time()
        # with torch.no_grad():
        if 1:
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
                if device == 'musa':
                    """
                    Declaration: aten::_scaled_dot_product_flash_attention_backward(
                    Tensor grad_out,  out[0]
                    Tensor query,  q
                    Tensor key,  k
                    Tensor value,  v
                    Tensor out,  out[0]
                    Tensor logsumexp, out[1]
                    Tensor cum_seq_q,  None
                    Tensor cum_seq_k, None
                    SymInt max_q, int
                    SymInt max_k, int
                    float dropout_p, 0.0
                    bool is_causal, 
                    Tensor philox_seed, 
                    Tensor philox_offset, *, float? scale=None) -> (Tensor grad_query, Tensor grad_key, Tensor grad_value)
                        """
                    grad_input = torch.ops.aten._scaled_dot_product_attention_flash_musa_backward(
                        attn_output[0],
                        # dump_grad,
                        q, k, v, #q, k, v,
                        *attn_output,
                        # *detached_ori_outputs, #(Tensor output, Tensor logsumexp, Tensor dropout_mask)
                        is_causal=True #causal same as fwd
                        )
                else:
                    grad_input = _flash_attn_backward(
                        dout = out,
                        # dump_grad,
                        q = q, 
                        k = k, 
                        v = v, #q, k, v,
                        out = out,
                        softmax_lse = softmax_lse,
                        dq = dq,
                        dk = dk,
                        dv = dv,
                        dropout_p=0.0,
                        softmax_scale = 1.0,
                        causal = True,
                        window_size=(-1,0),
                        alibi_slopes=None,
                        deterministic=False
                        # *detached_ori_outputs, #(Tensor output, Tensor logsumexp, Tensor dropout_mask)
                        # is_causal=True #causal same as fwd
                    ) 
    sync_device(device)
    bwd_elapsed_time = (time.time() - bwd_start_time) / (repeat - warmup) * 1000

    profiler.stop() if profiler else 1
    return fwd_elapsed_time, bwd_elapsed_time

def test(model, all_efficiency, qkv_contiguous, batch, seq_len, num_q_heads, num_kv_heads,  qk_head_dim, v_head_dim, TP=1, MAX_TFLOPS=None, res=None, device='musa'):
    assert num_q_heads % TP == 0 and num_kv_heads % TP == 0
    num_q_heads //= TP
    num_kv_heads //= TP
    shape_key = f'batch={batch}, seq_len={seq_len}, head_num={num_q_heads}, kv_head_num={num_kv_heads}, qk_head_dim={qk_head_dim}, v_head_dim={v_head_dim}, qkv_contiguous={qkv_contiguous}'
    if shape_key  in all_efficiency['sdp_fwd']['accurate_efficient_factor']:
        return
    query, key, value = generate_test_inputs(device, batch, seq_len,  num_q_heads, num_kv_heads, qk_head_dim, v_head_dim, qkv_contiguous)
    print(f"- 输入形状: query={query.shape}, key={key.shape}, value={value.shape}")
    print(f'batch: {batch}, seq_len: {seq_len}, num_heads: {num_q_heads}&{num_kv_heads}, head_dim: {qk_head_dim}')
    # 初始化配置和模块
    # 运行基准测试
    fwd_latency, bwd_latency = benchmark_flashattention(query, key, value, device = device)
    
    # 计算理论FLOPs (公式: 2 * batch * seq_len^2 * num_heads * head_dim)
    base_flops = batch * (seq_len ** 2) * max(num_q_heads, num_kv_heads)  * (qk_head_dim + v_head_dim)   
    fwd_flops = 2 * base_flops
    fwd_tflops = (fwd_flops / (fwd_latency * 1e-3+ 1e-12)) / 1e12  # 转换为TFLOPs

    bwd_flops = 5 * base_flops
    bwd_tflops = (bwd_flops / (bwd_latency * 1e-3+ 1e-12)) / 1e12  # 转换为TFLOPs
    
    print(f"=== {model} FlashAttention性能结果(TP={TP}):")
    print(f"- 延迟: {fwd_latency:.3f} ms/iteration, backward: {bwd_latency:.3f} ms/iteration")
    print(f"- fwd吞吐量: {fwd_tflops:.2f} TFLOPs, flops={fwd_flops} latency={fwd_latency:.2f} ms, 计算效率={fwd_tflops/MAX_TFLOPS:.2f}") 
    print(f"- bwd吞吐量: {bwd_tflops:.2f} TFLOPs, flops={bwd_flops} latency={bwd_latency:.2f} ms, 计算效率={bwd_tflops/MAX_TFLOPS:.2f}")
    print(f"- 输入形状: query={query.shape}, key={key.shape}, value={value.shape}")
    res['model'].append(model)
    res['TP'].append(TP)
    res['batch'].append(batch)
    res['seq_len'].append(seq_len)
    res['num_heads'].append(f'q={num_q_heads}, kv={num_kv_heads}')
    res['head_size'].append(f'qk={qk_head_dim//TP:.0f}, v={v_head_dim//TP:.0f}')
    res['flops'].append(f'fwd={fwd_flops}, bwd={bwd_flops}')
    res['time'].append(f'fwd={fwd_latency:.2f} ms, bwd={bwd_latency:.2f} ms')
    res['TFLOPS'].append(f'fwd={fwd_tflops:.2f} TFLOPS, bwd={bwd_tflops:.2f} TFLOPS')
    res['fwd_efficiency'].append(round(fwd_tflops/MAX_TFLOPS, 2))
    res['bwd_efficiency'].append(round(bwd_tflops/MAX_TFLOPS, 2))
    
    
    sdp_key = 'sdp_fwd'
    all_efficiency[sdp_key]['accurate_efficient_factor'][shape_key] = fwd_tflops/MAX_TFLOPS
    all_efficiency[sdp_key]['efficient_factor'] = sum(all_efficiency[sdp_key]['accurate_efficient_factor'].values()) / len(all_efficiency[sdp_key]['accurate_efficient_factor'])

    sdp_key = 'sdp_bwd'
    all_efficiency[sdp_key]['accurate_efficient_factor'][shape_key] = bwd_tflops/MAX_TFLOPS
    all_efficiency[sdp_key]['efficient_factor'] = sum(all_efficiency[sdp_key]['accurate_efficient_factor'].values()) / len(all_efficiency[sdp_key]['accurate_efficient_factor'])



# 5. 主测试流程
if __name__ == "__main__":
    # 准备输入数据
    system, device, MAX_TFLOPS = get_system_name()
    save_root =  f'{system}_fa_efficency'
    os.makedirs(save_root, exist_ok=True)

    res = {
        'model':[],
        'TP':[],
        'batch':[],
        'seq_len':[],
        'num_heads':[],
        'head_size':[],
        'flops':[],
        'TFLOPS':[],
        'time':[],
        'fwd_efficiency':[],
        'bwd_efficiency':[],
        # 'shape_str':[]
    }
    merged_dict = {
        'sdp_fwd':{
            'tflops': MAX_TFLOPS,
            'efficient_factor': 0,
            'accurate_efficient_factor':{
            }
        },
        'sdp_bwd':{
            'tflops': MAX_TFLOPS,
            'efficient_factor': 0,
            'accurate_efficient_factor':{
            }
        }    
    }
    test_params = {
        'ds_236b' :dict(batch=1, seq_len=4096, num_q_heads=128, num_kv_heads=128, qk_head_dim=192, v_head_dim=128, TP=1, MAX_TFLOPS=MAX_TFLOPS, res=res, device=device),
        'ds_v3' :dict(batch=1, seq_len=4096, num_q_heads=128, num_kv_heads=128, qk_head_dim=192, v_head_dim=128, TP=1, MAX_TFLOPS=MAX_TFLOPS, res=res, device=device),
        'llama3_8b': dict(batch=1, seq_len=4096, num_q_heads=32, num_kv_heads=8, qk_head_dim=128, v_head_dim=128, TP=1, MAX_TFLOPS=MAX_TFLOPS, res=res, device=device),
        'llama3_70b': dict(batch=1, seq_len=4096, num_q_heads=64, num_kv_heads=8, qk_head_dim=128, v_head_dim=128, TP=1, MAX_TFLOPS=MAX_TFLOPS, res=res, device=device),
        'qwen3_32b': dict(batch=1, seq_len=4096, num_q_heads=64, num_kv_heads=8, qk_head_dim=128, v_head_dim=128, TP=1, MAX_TFLOPS=MAX_TFLOPS, res=res, device=device),
    }
    for model, params in test_params.items():
        for tp in [1, 2, 4, 8]:
            params['TP'] = tp
            print(f"Running {model}...")
            
            # test(model, merged_dict, True, **params)
            # if device == 'musa':
            test(model, merged_dict, False, **params)
    
    import pandas as pd
    import json
    df = pd.DataFrame(res)
    df.to_csv(os.path.join(save_root, 'fa_efficency_test.csv'), index=False)
    with open(os.path.join(save_root, 'fa_efficency_test.json'), 'w') as f:
        json.dump(merged_dict, f, indent=4)
