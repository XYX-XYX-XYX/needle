import sys 
sys.path.append('./python')
import argparse
import numpy as np
import needle as ndl
from needle import backend_ndarray as nd

def benchmark_flash_attention(batch_size, num_heads, seq_len, head_dim, causal, dropout, warmup=10, repeats=100):
    device = ndl.cuda()
    if not device.enabled():
        print("CUDA device not available. Profiling skipped.")
        return

    print(f"Benchmarking Flash Attention: B={batch_size}, H={num_heads}, L={seq_len}, D={head_dim}, "
          f"Causal={causal}, Dropout={dropout}")
    
    # 构造数据
    shape = (batch_size, num_heads, seq_len, head_dim)
    q_np = np.random.randn(*shape).astype(np.float16)
    k_np = np.random.randn(*shape).astype(np.float16)
    v_np = np.random.randn(*shape).astype(np.float16)
    
    q_ndl = ndl.Tensor(q_np, device=device)
    k_ndl = ndl.Tensor(k_np, device=device)
    v_ndl = ndl.Tensor(v_np, device=device)

    if not hasattr(device, "flash_attention_forward_benchmark"):
        raise RuntimeError(
            "CUDA backend does not expose flash_attention_forward_benchmark; "
            "rebuild with make lib."
        )

    q_arr = q_ndl.realize_cached_data().compact()
    k_arr = k_ndl.realize_cached_data().compact()
    v_arr = v_ndl.realize_cached_data().compact()
    out_arr = nd.empty(shape, device=device)

    def run_backend_benchmark(num_iters):
        return device.flash_attention_forward_benchmark(
            q_arr._handle,
            k_arr._handle,
            v_arr._handle,
            out_arr._handle,
            batch_size,
            num_heads,
            seq_len,
            seq_len,
            head_dim,
            dropout,
            causal,
            num_iters,
        )

    # 1. Warm-up
    if warmup > 0:
        print(f"Warming up for {warmup} iterations...")
        run_backend_benchmark(warmup)

    # 2. Benchmark
    print(f"Running {repeats} iterations...")
    elapsed_time_ms = run_backend_benchmark(repeats)
    avg_latency_ms = elapsed_time_ms / repeats

    print(f"Elapsed Time: {elapsed_time_ms:.6f} ms")
    print(f"Average Latency: {avg_latency_ms:.6f} ms")
    
    # 计算 TFLOPS (近似公式)
    # FlashAttn FLOPs approx: 4 * B * H * L^2 * D (Attention calculation)
    flops = 4 * batch_size * num_heads * (seq_len ** 2) * head_dim
    if causal:
        flops /= 2

    tflops = (flops / (avg_latency_ms / 1000)) / 1e12
    print(f"Approx. FLOPs per iter: {flops / 1e9:.3f} GFLOPs")
    print(f"Approx. TFLOPS: {tflops:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()

    benchmark_flash_attention(
        args.batch_size, args.num_heads, args.seq_len, 
        args.head_dim, args.causal, args.dropout, args.warmup, args.repeats
    )
