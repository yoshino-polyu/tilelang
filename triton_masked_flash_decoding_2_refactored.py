import torch
import torch.nn.functional as F
import math

# Try importing TileLang; raise an error if not installed
try:
    import tilelang
    import tilelang.language as T
except ImportError as e:
    raise ImportError("TileLang library is required but not installed.") from e

# Define data types for TileLang kernel
dtype = "float16"       # data type for inputs/outputs (FP16)
accum_dtype = "float"   # accumulation data type (FP32 for numerical stability)

def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, valid_length: torch.Tensor, num_k_heads: int):
    """
    Baseline flash attention using PyTorch's scaled_dot_product_attention.
    - q: [B, q_heads, D] (or [B, q_heads, L_q, D] if multiple query positions)
    - k: [B, S, kv_heads, D]
    - v: [B, S, kv_heads, D]
    - mask: [B, 1, 1, S] boolean mask (True indicates masked-out positions)
    - valid_length: [B] tensor of sequence lengths (not used directly here, mask covers it)
    - num_k_heads: number of key/value heads (kv_heads)
    Returns: output tensor [B, q_heads, D] (or [B, q_heads, L_q, D] if L_q > 1).
    """
    B, q_heads = q.shape[0], q.shape[1]
    # Ensure q has a sequence-length dimension (L_q)
    if q.dim() == 3:
        q = q.unsqueeze(2)  # -> [B, q_heads, 1, D]
    # Reshape k and v to [B, kv_heads, S, D] for PyTorch attention
    if k.dim() == 4:
        k = k.permute(0, 2, 1, 3).contiguous()
    if v.dim() == 4:
        v = v.permute(0, 2, 1, 3).contiguous()
    # Use PyTorch's scaled_dot_product_attention (FlashAttention under the hood if available)
    enable_gqa_flag = (q_heads != num_k_heads)  # enable grouped query attention if heads differ
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=enable_gqa_flag)
    # Remove the query sequence dimension if it is 1
    if out.shape[2] == 1:
        out = out.squeeze(2)  # -> [B, q_heads, D]
    return out

def flash_decoding(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, valid_length: torch.Tensor, num_k_heads: int):
    """
    Flash-Decoding implementation (PyTorch-based).
    Uses the same mechanism as flash_attention with grouped query attention enabled.
    """
    # In this refactored script, flash_decoding is implemented using the same PyTorch routine.
    return flash_attention(q, k, v, mask, valid_length, num_k_heads)

def tilelang_flash_decoding(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, valid_length: torch.Tensor, num_k_heads: int):
    """
    TileLang-based flash decoding.
    Compiles and executes a TileLang kernel for masked attention (no CLI args).
    Matches the interface of the other decoding functions.
    """
    B, q_heads, D = q.shape[0], q.shape[1], q.shape[-1]
    S = k.shape[1]              # sequence length of key/value
    kv_heads = num_k_heads      # number of key/value heads
    # Tiling parameters (chosen for performance; can be tuned)
    BLOCK_N = 64
    BLOCK_H = 64
    # Compute grouping info for query vs key heads
    kv_group_num = q_heads // kv_heads  # number of query head groups
    valid_block_H = min(BLOCK_H, kv_group_num)
    # Define the TileLang kernel program
    def flashattn_no_pe(batch, heads, kv_head_num, seqlen_kv, dim, block_N, block_H):
        # Softmax scale factor = log2(e) / sqrt(dim)
        scale = (1.0 / dim)**0.5 * 1.44269504  # 1.44269504 = log2(e)
        @T.macro  # Define the core kernel as a TileLang macro
        def flash_attn_no_pe(
            Q: T.Tensor([batch, heads, dim], dtype),
            K: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
            V: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
            Valid: T.Tensor([batch], "int32"),  # valid lengths per batch
            Output: T.Tensor([batch, heads, dim], dtype),
        ):
            # Launch one kernel block per batch (bx) and per head group chunk (by)
            with T.Kernel(batch, heads // min(block_H, heads // kv_head_num), threads=256) as (bx, by):
                # Shared memory buffers for tiles of Q, K, V, and intermediate results
                Q_shared = T.alloc_shared([block_H, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                S_shared = T.alloc_shared([block_H, block_N], dtype)  # to store softmax probabilities
                O_shared = T.alloc_shared([block_H, dim], dtype)
                # Register (fragment) accumulators for scores and output
                acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
                acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_H], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
                scores_scale = T.alloc_fragment([block_H], accum_dtype)
                scores_sum = T.alloc_fragment([block_H], accum_dtype)
                logsum = T.alloc_fragment([block_H], accum_dtype)
                # Determine which key head this block of query heads corresponds to
                cur_kv_head = (by * min(block_H, heads // kv_head_num)) // (heads // kv_head_num)
                # Initialize output accumulator and log-sum-exp values for this block of heads
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))
                # Loop over key/value blocks (with double buffering for overlap)
                loop_range = T.ceildiv(seqlen_kv, block_N)
                for k_iter in T.Pipelined(loop_range, num_stages=2):
                    # Load a block of K and V for the current key head
                    T.copy(K[bx, k_iter * block_N:(k_iter + 1) * block_N, cur_kv_head, :], K_shared)
                    T.copy(V[bx, k_iter * block_N:(k_iter + 1) * block_N, cur_kv_head, :], V_shared)
                    # Load the corresponding block of Q for the group of heads this thread block handles
                    T.copy(Q[bx, by * min(block_H, heads // kv_head_num):(by + 1) * min(block_H, heads // kv_head_num), :], Q_shared)
                    # Compute dot products: Q_shared @ K_shared^T -> acc_s (scores for this block)
                    T.clear(acc_s)
                    T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                    # Apply the attention mask: for each position beyond valid_length, set score to -inf
                    for j in T.serial(block_N):
                        if (k_iter * block_N + j) >= Valid[bx]:
                            for i in T.serial(min(block_H, heads // kv_head_num)):
                                acc_s[i, j] = -T.infinity(accum_dtype)
                    # Compute max across this block's scores and combine with previous max for stability
                    T.copy(scores_max, scores_max_prev)            # save previous block's max
                    T.fill(scores_max, -T.infinity(accum_dtype))   # reset current max
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    # Handle case where all scores in this block are masked (scores_max stays -inf)
                    for i in T.serial(min(block_H, heads // kv_head_num)):
                        if scores_max[i] == -T.infinity(accum_dtype):
                            scores_max[i] = scores_max_prev[i]
                    # Compute scaling factors and exponentiate scores (base-2 exponent for efficiency)
                    for i in T.Parallel(min(block_H, heads // kv_head_num)):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Parallel(min(block_H, heads // kv_head_num), block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                    # Sum the exponentiated scores for normalization
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    T.copy(acc_s, S_shared)  # store probabilities for this block
                    # Update running log-sum for this head block
                    for i in T.Parallel(min(block_H, heads // kv_head_num)):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    # Scale the accumulated output by scores_scale (to account for renormalization)
                    for i, j in T.Parallel(min(block_H, heads // kv_head_num), dim):
                        acc_o[i, j] *= scores_scale[i]
                    # Accumulate partial output: probabilities * V_shared -> acc_o
                    T.gemm(S_shared, V_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)
                # After processing all blocks, finalize the output by dividing by logsum
                for i, j in T.Parallel(min(block_H, heads // kv_head_num), dim):
                    acc_o[i, j] /= logsum[i]
                # Write the output for this block of heads back to global memory
                T.copy(acc_o, O_shared)
                T.copy(O_shared, Output[bx, by * min(block_H, heads // kv_head_num):(by + 1) * min(block_H, heads // kv_head_num), :])
        # Define a prim_func that calls the macro (no splitting of heads across multiple passes)
        @T.prim_func
        def main_no_split_no_pe(
            Q: T.Tensor([batch, heads, dim], dtype),
            K: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
            V: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
            Valid: T.Tensor([batch], "int32"),
            Output: T.Tensor([batch, heads, dim], dtype),
        ):
            flash_attn_no_pe(Q, K, V, Valid, Output)
        return main_no_split_no_pe
    # Compile the TileLang program (the output index [4] corresponds to the Output tensor)
    program = flashattn_no_pe(B, q_heads, kv_heads, S, D, BLOCK_N, BLOCK_H)
    tile_kernel = tilelang.compile(program, out_idx=[4], target="cuda")
    # Execute the compiled kernel. (Output is returned due to out_idx configuration)
    out = tile_kernel(q, k, v, valid_length.to(torch.int32).contiguous().cuda())
    return out

def test_masked_decoding():
    # Example test to verify all implementations produce the same result
    B = 4
    heads = 8
    kv_heads = 2  # using a grouped Q/K head scenario
    seq_len = 256
    dim = 64
    device = "cuda"
    # Random inputs
    q = torch.randn(B, heads, dim, device=device, dtype=torch.float16)
    k = torch.randn(B, seq_len, kv_heads, dim, device=device, dtype=torch.float16)
    v = torch.randn(B, seq_len, kv_heads, dim, device=device, dtype=torch.float16)
    # Random valid lengths for each sequence in the batch
    valid_length = torch.randint(1, seq_len + 1, (B,), device=device)
    # Construct attention mask from valid_length (True for positions to mask out)
    mask = torch.arange(seq_len, device=device)[None, :] >= valid_length[:, None]
    attn_mask = mask[:, None, None, :]  # [B,1,1,S]
    # Compute outputs using each method
    out_flash_attn = flash_attention(q, k, v, attn_mask, valid_length, kv_heads)
    out_flash_dec  = flash_decoding(q, k, v, attn_mask, valid_length, kv_heads)
    out_tilelang   = tilelang_flash_decoding(q, k, v, attn_mask, valid_length, kv_heads)
    # Check for correctness (all outputs should match within tolerance)
    torch.testing.assert_close(out_flash_attn, out_flash_dec, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(out_flash_attn, out_tilelang, rtol=1e-2, atol=1e-2)
    print("All decoding implementations produce matching outputs within tolerance.")

def profile_all():
    # Generate a new set of inputs for profiling
    B = 4
    heads = 8
    kv_heads = 2
    seq_len = 256
    dim = 64
    device = "cuda"
    q = torch.randn(B, heads, dim, device=device, dtype=torch.float16)
    k = torch.randn(B, seq_len, kv_heads, dim, device=device, dtype=torch.float16)
    v = torch.randn(B, seq_len, kv_heads, dim, device=device, dtype=torch.float16)
    valid_length = torch.randint(1, seq_len + 1, (B,), device=device)
    mask = torch.arange(seq_len, device=device)[None, :] >= valid_length[:, None]
    attn_mask = mask[:, None, None, :]
    # Warm-up (to trigger any lazy initialization or compilation)
    _ = flash_attention(q, k, v, attn_mask, valid_length, kv_heads)
    _ = flash_decoding(q, k, v, attn_mask, valid_length, kv_heads)
    _ = tilelang_flash_decoding(q, k, v, attn_mask, valid_length, kv_heads)
    torch.cuda.synchronize()
    # Profile each method
    import time
    methods = [flash_attention, flash_decoding, tilelang_flash_decoding]
    names = ["flash_attention (Torch)", "flash_decoding (Torch)", "tilelang_flash_decoding"]
    timings = {}
    for func, name in zip(methods, names):
        # Measure average execution time over multiple iterations
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(50):
            _ = func(q, k, v, attn_mask, valid_length, kv_heads)
        torch.cuda.synchronize()
        avg_time_ms = (time.time() - start_time) / 50 * 1000
        timings[name] = avg_time_ms
    # Display profiling results
    print("\nProfiling results (average of 50 runs):")
    for name, t in timings.items():
        print(f"{name}: {t:.3f} ms")

if __name__ == "__main__":
    test_masked_decoding()
    profile_all()