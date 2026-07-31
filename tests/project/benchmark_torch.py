import torch
import torch.nn.functional as F
import argparse
import contextlib
import inspect


def get_cudnn_version():
    """Return the cuDNN runtime version as a readable string and raw integer."""
    version = torch.backends.cudnn.version()
    if version is None:
        return "unavailable", None

    # cuDNN 9 uses M * 10000 + m * 100 + p; older versions use
    # M * 1000 + m * 100 + p.
    if version >= 90000:
        major, remainder = divmod(version, 10000)
    else:
        major, remainder = divmod(version, 1000)
    minor, patch = divmod(remainder, 100)
    return f"{major}.{minor}.{patch}", version


def make_sdpa_context(backend: str):
    """
    backend:
      - flash: PyTorch built-in FLASH_ATTENTION backend, FA2-like path
      - cudnn: cuDNN SDPA backend
      - mem_efficient: memory-efficient attention backend
      - math: PyTorch C++ math backend
      - auto: let PyTorch choose automatically
    """
    backend = backend.lower()

    if backend == "auto":
        return contextlib.nullcontext()

    # New recommended API: torch.nn.attention.sdpa_kernel
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend

        mapping = {
            "flash": SDPBackend.FLASH_ATTENTION,
            "cudnn": SDPBackend.CUDNN_ATTENTION,
            "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
            "math": SDPBackend.MATH,
        }

        if backend not in mapping:
            raise ValueError(f"Unknown backend: {backend}")

        return sdpa_kernel(mapping[backend])

    except ImportError:
        # Fallback for older PyTorch API
        from torch.backends.cuda import sdp_kernel

        flags = {
            "enable_flash": False,
            "enable_math": False,
            "enable_mem_efficient": False,
        }

        if backend == "flash":
            flags["enable_flash"] = True
        elif backend == "math":
            flags["enable_math"] = True
        elif backend == "mem_efficient":
            flags["enable_mem_efficient"] = True
        elif backend == "cudnn":
            pass
        else:
            raise ValueError(f"Unknown backend: {backend}")

        # Newer old-API also has enable_cudnn=True/False.
        # For backend=flash, explicitly disable cuDNN so it cannot steal the run.
        try:
            sig = inspect.signature(sdp_kernel)
            if "enable_cudnn" in sig.parameters:
                flags["enable_cudnn"] = backend == "cudnn"
            elif backend == "cudnn":
                raise RuntimeError(
                    "This PyTorch version does not expose enable_cudnn in "
                    "torch.backends.cuda.sdp_kernel."
                )
        except (TypeError, ValueError):
            if backend == "cudnn":
                raise RuntimeError(
                    "Cannot verify cuDNN SDPA support from this PyTorch version."
                )

        return sdp_kernel(**flags)


def profile_backend_once(q, k, v, causal, dropout):
    """
    Optional profiler check.
    Look for:
      aten::_scaled_dot_product_flash_attention
      aten::_scaled_dot_product_cudnn_attention
      aten::_scaled_dot_product_efficient_attention
      aten::_scaled_dot_product_attention_math
    """
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:
        for _ in range(5):
            F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=dropout,
                is_causal=causal,
            )

    torch.cuda.synchronize()

    print("\n===== Profiler Backend Check =====")
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=30,
    ))
    print("==================================\n")


def benchmark_torch_flash(
    batch_size,
    num_heads,
    seq_len,
    head_dim,
    causal,
    dropout,
    backend="flash",
    dtype_str="fp16",
    warmup=10,
    repeats=100,
    profile=False,
):
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping.")
        return

    backend = backend.lower()
    device = torch.device("cuda")

    if dtype_str == "fp16":
        dtype = torch.float16
    elif dtype_str == "bf16":
        dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

    shape = (batch_size, num_heads, seq_len, head_dim)

    q = torch.randn(shape, device=device, dtype=dtype)
    k = torch.randn(shape, device=device, dtype=dtype)
    v = torch.randn(shape, device=device, dtype=dtype)

    torch.cuda.synchronize()

    if backend == "cudnn":
        cudnn_version, cudnn_version_raw = get_cudnn_version()
        if cudnn_version_raw is None:
            print("cuDNN version: unavailable")
        else:
            print(
                f"cuDNN version: {cudnn_version} "
                f"(torch.backends.cudnn.version()={cudnn_version_raw})"
            )

    print(
        f"Benchmarking PyTorch SDPA: "
        f"backend={backend}, "
        f"B={batch_size}, H={num_heads}, "
        f"L={seq_len}, D={head_dim}, "
        f"Causal={causal}, Dropout={dropout}, Dtype={dtype}"
    )

    with torch.inference_mode():
        with make_sdpa_context(backend):
            # Optional: verify actual aten op / kernel path
            if profile:
                profile_backend_once(q, k, v, causal, dropout)

            print(f"Warming up for {warmup} iterations...")
            for _ in range(warmup):
                F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=dropout,
                    is_causal=causal,
                )

            torch.cuda.synchronize()

            print(f"Running {repeats} iterations...")
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            for _ in range(repeats):
                F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=dropout,
                    is_causal=causal,
                )
            end_event.record()

            torch.cuda.synchronize()

            elapsed_time_ms = start_event.elapsed_time(end_event)
            avg_latency_ms = elapsed_time_ms / repeats

    print(f"Average Latency: {avg_latency_ms:.6f} ms")

    # Forward FlashAttention FLOPs approximation:
    # QK^T: 2 * B * H * L * L * D
    # PV:   2 * B * H * L * L * D
    # total = 4 * B * H * L^2 * D
    flops_per_iter = 4 * batch_size * num_heads * (seq_len ** 2) * head_dim

    if causal:
        flops_per_iter /= 2

    avg_latency_s = avg_latency_ms / 1000.0
    tflops = (flops_per_iter / 1e12) / avg_latency_s

    print(f"Approx. FLOPs per iter: {flops_per_iter / 1e9:.3f} GFLOPs")
    print(f"Approx. TFLOPS: {tflops:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--head_dim", type=int, default=64)

    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument(
        "--backend",
        type=str,
        default="flash",
        choices=["flash", "cudnn", "mem_efficient", "math", "auto"],
        help=(
            "SDPA backend. "
            "flash = PyTorch built-in FLASH_ATTENTION backend, FA2-like. "
            "cudnn = cuDNN SDPA backend. "
            "mem_efficient = memory-efficient attention. "
            "math = PyTorch C++ math backend. "
            "auto = let PyTorch choose."
        ),
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16"],
    )

    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)

    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run torch.profiler once to verify actual SDPA backend op.",
    )

    args = parser.parse_args()

    benchmark_torch_flash(
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        causal=args.causal,
        dropout=args.dropout,
        backend=args.backend,
        dtype_str=args.dtype,
        warmup=args.warmup,
        repeats=args.repeats,
        profile=args.profile,
    )
