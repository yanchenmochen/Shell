import torch
import torch_musa

q = torch.load("q.pt", map_location='musa:0', weights_only=False)
k = torch.load("k.pt", map_location='musa:0', weights_only=False)
v = torch.load("v.pt", map_location='musa:0', weights_only=False)
# print(q)
print(torch.isnan(q).any())
print(torch.isnan(k).any())
print(torch.isnan(v).any())
bs = q.shape[0]
q_seq_len = q.shape[1]
head_dim= q.shape[-1]
kv_seq_len = k.shape[1]
with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
    attn_output = torch.nn.functional.scaled_dot_product_attention(
            q.view(bs, q_seq_len, q.shape[-2], q.shape[-1]).transpose(1, 2),
            k.view(bs, kv_seq_len, k.shape[-2], k.shape[-1]).transpose(1, 2),
            v.view(bs, kv_seq_len, v.shape[-2], v.shape[-1]).transpose(1, 2),
            dropout_p=0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=True #self.is_causal and attention_mask is None and q_len > 1,
        )
    print(attn_output.shape)
    
    print(torch.isnan(attn_output).any())
    print(torch.isnan(attn_output.transpose(1, 2).contiguous().view(bs, q_seq_len, q.shape[-2], v.shape[-1])).any())