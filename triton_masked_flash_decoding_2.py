import torch
from flash_attn import flash_attn_func, flash_attn_with_kvcache

def profile(func, inputs, num_warmups=100, num_iters=100):
    torch.cuda.synchronize()
    for _ in range(num_warmups):
        func(*inputs)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        func(*inputs)
    end.record()
    torch.cuda.synchronize()
    latency = start.elapsed_time(end) / num_iters
    return latency

def flash_attention(
    q: torch.Tensor,  # [bsz, num_q_heads, qlen(=1), dim]
    k: torch.Tensor,  # [bsz, num_k_heads, max_length, dim]
    v: torch.Tensor,  # [bsz, num_k_heads, max_length, dim]
    mask: torch.Tensor,  # [bsz, num_k_heads, max_length]
    valid_length: int,
    num_k_heads: int,
):
    bsz, num_q_heads, _, dim = q.shape
    k = k[:bsz*num_k_heads*valid_length*dim].view(bsz, num_k_heads, valid_length, dim)
    v = v[:bsz*num_k_heads*valid_length*dim].view(bsz, num_k_heads, valid_length, dim)
    attn_out, attn_lse, _ = flash_attn_func(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        return_attn_probs=True,
    )
    return attn_lse, attn_out.transpose(1, 2)


def flash_decoding(
    q: torch.Tensor,  # [bsz, num_q_heads, qlen(=1), dim]
    k: torch.Tensor,  # [bsz, num_k_heads, max_length, dim]
    v: torch.Tensor,  # [bsz, num_k_heads, max_length, dim]
    mask: torch.Tensor,  # [bsz, num_k_heads, max_length]
    valid_length: int,
    num_k_heads: int,
):
    bsz, num_q_heads, _, dim = q.shape
    k = k[:bsz*num_k_heads*valid_length*dim].view(bsz, num_k_heads, valid_length, dim)
    v = v[:bsz*num_k_heads*valid_length*dim].view(bsz, num_k_heads, valid_length, dim)
    attn_out, attn_lse = flash_attn_with_kvcache(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        return_softmax_lse=True,
    )
    return attn_lse, attn_out.transpose(1, 2)


def profile_all(funcs, inputs):
    results = {}
    max_tag_length = max(map(len, funcs.keys()))
    for tag, func in funcs.items():
        latency = profile(func, inputs)
        results[tag] = latency
        print(f'{tag:{max_tag_length}} : {latency:5.3f} ms')
    return results

def test_masked_decoding(
    bsz: int,
    num_k_heads: int,
    num_q_heads: int,
    max_length: int,
    dim: int,
    device: torch.device = 'cuda',
    dtype: torch.dtype = torch.float16,
):
    # print(f'bsz={bsz}, num_k_heads={num_k_heads}, num_q_heads={num_q_heads}, max_length={max_length}, dim={dim}')
    q = torch.randn((bsz, num_q_heads, 1, dim), dtype=dtype, device=device)
    k = torch.randn((bsz*num_k_heads*max_length*dim, ), dtype=dtype, device=device)
    v = torch.randn((bsz*num_k_heads*max_length*dim, ), dtype=dtype, device=device)
    mask = torch.randint(0, 2, (bsz*num_k_heads*max_length, ), dtype=torch.bool, device=device)
    valid_length = max_length // 2
    inputs = [q, k, v, mask, valid_length, num_k_heads]
    
    return profile_all({
        'flash_attn': flash_attention,
        'flash_decoding': flash_decoding,
    }, inputs)


if __name__ == '__main__':
    from tqdm import tqdm
    import pandas as pd
    df = []
    for bsz in tqdm([1, 2, 4, 8]):
        results = test_masked_decoding(
            bsz=bsz,
            num_k_heads=8,
            num_q_heads=32,
            max_length=131072,
            dim=128,
        )
        df.append(pd.DataFrame({bsz: results}).T)
    df = pd.concat(df).round(3)
    df.index.name = 'bsz'
    print(df)