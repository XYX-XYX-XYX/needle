import torch
import torch.nn.functional as F
import time
import argparse
import math

def benchmark_torch_flash(batch_size, num_heads, seq_len, head_dim, causal, dropout, warmup=10, repeats=100):
    # 1. 检查 CUDA 设备
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping.")
        return
    
    device = torch.device("cuda")
    
    # 2. 准备数据 (必须是 fp16 或 bf16，因为 FlashAttention 不支持 fp32)
    dtype = torch.float16  # 或者 torch.bfloat16 (A100/H100 推荐)
    
    # PyTorch SDPA 支持的形状通常是 (B, H, L, D) 或者 (B, L, H, D)
    # 这里保持和你的 needle 实现一致: (B, H, L, D)
    shape = (batch_size, num_heads, seq_len, head_dim)
    
    q = torch.randn(shape, device=device, dtype=dtype)
    k = torch.randn(shape, device=device, dtype=dtype)
    v = torch.randn(shape, device=device, dtype=dtype)

    print(f"Benchmarking PyTorch SDPA (FlashAttention): B={batch_size}, H={num_heads}, "
          f"L={seq_len}, D={head_dim}, Causal={causal}, Dropout={dropout}, Dtype={dtype}")

    # 3. 强制使用 FlashAttention 后端
    # 这段上下文管理器确保我们测量的确实是 FlashAttention，而不是 math 或 memory-efficient
    # 注意：这需要 PyTorch 2.0+
    try:
        from torch.backends.cuda import sdp_kernel
        ctx = sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)
    except ImportError:
        print("Warning: torch.backends.cuda.sdp_kernel not found. "
              "Ensure you are using PyTorch 2.0+. Running without explicit backend enforcement.")
        ctx = torch.no_grad() # 空上下文作为 fallback

    # 4. Benchmark 流程
    with ctx:
        # --- Warmup ---
        print(f"Warming up for {warmup} iterations...")
        for _ in range(warmup):
            F.scaled_dot_product_attention(q, k, v, dropout_p=dropout, is_causal=causal)
        torch.cuda.synchronize() # 确保 Warmup 完成

        # --- Timing ---
        print(f"Running {repeats} iterations...")
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(repeats):
            F.scaled_dot_product_attention(q, k, v, dropout_p=dropout, is_causal=causal)
        end_event.record()

        # 等待结束
        torch.cuda.synchronize()
        
        # 计算时间 (ms)
        elapsed_time_ms = start_event.elapsed_time(end_event)
        avg_latency_ms = elapsed_time_ms / repeats

    print(f"Average Latency: {avg_latency_ms:.3f} ms")

    # 5. 计算 TFLOPS
    # 公式：4 * B * H * L^2 * D
    flops_per_iter = 4 * batch_size * num_heads * (seq_len ** 2) * head_dim
    
    if causal:
        flops_per_iter /= 2

    # TFLOPS = (FLOPs / 10^12) / (Time_in_seconds)
    avg_latency_s = avg_latency_ms / 1000.0
    tflops = (flops_per_iter / 1e12) / avg_latency_s

    print(f"Approx. TFLOPS: {tflops:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.0)
    args = parser.parse_args()

    benchmark_torch_flash(
        args.batch_size, args.num_heads, args.seq_len, 
        args.head_dim, args.causal, args.dropout
    )