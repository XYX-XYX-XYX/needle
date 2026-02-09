import sys 
sys.path.append('./python')
import time
import argparse
import numpy as np
import needle as ndl
import needle.ops as ops
import torch

def benchmark_flash_attention(batch_size, num_heads, seq_len, head_dim, causal, dropout, warmup=10, repeats=100):
    device = ndl.cuda()
    if not device.enabled():
        print("CUDA device not available. Profiling skipped.")
        return

    print(f"Benchmarking Flash Attention: B={batch_size}, H={num_heads}, L={seq_len}, D={head_dim}, "
          f"Causal={causal}, Dropout={dropout}")
    
    # 构造数据
    shape = (batch_size, num_heads, seq_len, head_dim)
    q_np = np.random.randn(*shape).astype(np.float32)
    k_np = np.random.randn(*shape).astype(np.float32)
    v_np = np.random.randn(*shape).astype(np.float32)
    
    q_ndl = ndl.Tensor(q_np, device=device)
    k_ndl = ndl.Tensor(k_np, device=device)
    v_ndl = ndl.Tensor(v_np, device=device)

    # 为了确保测量准确，最好加上 device.synchronize()，但 Needle 目前可能只支持隐式同步
    # 如果您的 NDArray 后端操作是异步的（cuda通常是），则必须在计时前后同步。
    # 假设 NDArray.numpy() 会强制同步。或者调用 backend 特定的 sync。
    # 由于 Needle 教学代码通常简化了同步，我们这里假设 compute 是同步的或者我们通过 .numpy() 强制同步。
    
    # 1. Warm-up
    print(f"Warming up for {warmup} iterations...")
    for _ in range(warmup):
        out = ops.flashattention(q_ndl, k_ndl, v_ndl, dropout=dropout, causal=causal)
        # 强制同步：对于 CUDA，最简单的方法通常是把结果拷回 CPU
        _ = out.numpy() 

    # 2. Benchmark
    print(f"Running {repeats} iterations...")
    _ = out.numpy() 
    # 或者如果 needle backend 有 sync 接口: device.backend.synchronize()
    
    start_time = time.time()
    for _ in range(repeats):
        out = ops.flashattention(q_ndl, k_ndl, v_ndl, dropout=dropout, causal=causal)
        # 注意：如果在测试纯 kernel 耗时，不要在循环里 .numpy()，那会测到数据传输时间
        # 但如果不 .numpy()，异步执行可能导致计时不准。
        # 标准做法：循环运行 kernel -> 循环外最后一次 synchronize
    
    # 等待所有 CUDA 任务完成
    # 由于 Needle 作业可能没有显式的 sync API，我们用 torch.cuda.synchronize() 来蹭一下
    # 或者做一个极小的 .numpy() 操作来强制同步
    # _ = out.numpy() # 这会包含最后一次的数据传输时间，但对于 repeats=100 影响较小
    
    # 最稳妥的方式：
    _ = out.numpy() 
    end_time = time.time()

    total_time = end_time - start_time
    avg_slatency = (total_time / repeats) * 1000 # ms

    print(f"Average Latency: {avg_slatency:.3f} ms")
    
    # 计算 TFLOPS (近似公式)
    # FlashAttn FLOPs approx: 4 * B * H * L^2 * D (Attention calculation)
    # 真正的 flops 还要考虑 causal masking 等
    flops = 4 * batch_size * num_heads * (seq_len ** 2) * head_dim
    tflops = (flops / (avg_slatency / 1000)) / 1e12
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

    benchmark_flash_attention(
        args.batch_size, args.num_heads, args.seq_len, 
        args.head_dim, args.causal, args.dropout
    )