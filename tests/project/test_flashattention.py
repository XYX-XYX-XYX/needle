import sys 
sys.path.append('./python')
import numpy as np
import pytest
import torch
import itertools
import os
import needle as ndl
import needle.nn as nn

np.random.seed(3)

CUDA_DEVICE = pytest.param(
    ndl.cuda(),
    marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU"),
    id="cuda",
)


def _cuda_compute_capability():
    capability = getattr(ndl.cuda(), "__cuda_compute_capability__", (-1, -1))
    return tuple(capability)


def _skip_unsupported_kernel(kernel):
    major, _ = _cuda_compute_capability()
    if kernel == "sm90" and major < 9:
        pytest.skip("sm90 FlashAttention kernel requires compute capability 9.x")
    if kernel == "sm80" and major < 8:
        pytest.skip("sm80 FlashAttention kernel requires compute capability 8.x")


def _resolve_backend_kernel(device, kernel):
    if hasattr(device, "flash_attention_resolve_kernel"):
        return device.flash_attention_resolve_kernel(kernel)

    major, _ = _cuda_compute_capability()
    if kernel == "auto":
        return "sm90" if major >= 9 else "sm80"
    return kernel


# @pytest.mark.parametrize("batch_size", [4, 8])
# @pytest.mark.parametrize("num_heads", [5])
# @pytest.mark.parametrize("queries_len", [31])
# @pytest.mark.parametrize("inner_dim", [64])
# @pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize("dropout", [0.0, 0.1])
# @pytest.mark.parametrize("device", _DEVICES, ids=["cuda"])
# def test_flashattention_activation(batch_size, num_heads, queries_len, inner_dim, causal, dropout, device):

#     np.random.seed(19943)

#     q = np.random.randn(
#         batch_size, num_heads,
#         queries_len, inner_dim).astype(np.float32)

#     layer = nn.FlashMutiHeadAttention(
#         dropout=dropout, causal=causal, device=device)

#     result, probs = layer(
#         ndl.Tensor(q, device=device),
#         ndl.Tensor(q, device=device),
#         ndl.Tensor(q, device=device),
#     )

#     probs = probs.numpy()

#     current_input_id = "-".join([str(x) for x in (
#         batch_size, num_heads, queries_len, inner_dim, causal, dropout, device
#     )])

#     labels_path = (
#         "./tests/hw4/data/" + 
#         "test_attention_activation-{}.npy"
#         .format(current_input_id))

#     with open(labels_path, 'rb') as f:
#         label_probs = np.load(f)

#     # np.testing.assert_array_equal(probs, label_probs)
#     np.testing.assert_allclose(probs, label_probs, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("batch_size", [2, 16])
@pytest.mark.parametrize("num_heads", [5, 10, 32])
@pytest.mark.parametrize("queries_len", [128, 256, 1024, 2048])
@pytest.mark.parametrize("inner_dim", [64])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dropout", [0.0]) # 为对比数值必须为0
@pytest.mark.parametrize("kernel", ["auto", "sm80", "sm90"])
@pytest.mark.parametrize("device", [CUDA_DEVICE])
def test_attention_activation_vs_torch(batch_size, num_heads, queries_len, inner_dim, causal, dropout, kernel, device, capsys):
    _skip_unsupported_kernel(kernel)
    resolved_kernel = _resolve_backend_kernel(device, kernel)
    with capsys.disabled():
        print(
            "[FlashAttention test] "
            f"requested_kernel={kernel}, resolved_kernel={resolved_kernel}, "
            f"batch_size={batch_size}, num_heads={num_heads}, "
            f"seq_len={queries_len}, head_dim={inner_dim}, "
            f"causal={causal}, dropout={dropout}, "
            f"device={device}, compute_capability={_cuda_compute_capability()}",
            flush=True,
        )
    # Skip non-CUDA if using flash attention, but standard attention works on CPU
    # if device == ndl.cpu(): ... 
    
    np.random.seed(19943)
    
    # 构造Q, K, V数据 (Match default Needle MultiHeadAttention shape: B, H, L, D)
    q_np = np.random.randn(batch_size, num_heads, queries_len, inner_dim).astype(np.float16)
    k_np = np.random.randn(batch_size, num_heads, queries_len, inner_dim).astype(np.float16)
    v_np = np.random.randn(batch_size, num_heads, queries_len, inner_dim).astype(np.float16)

    # --- Needle 运行 ---
    # MultiHeadAttention 的输入就是 (B, H, L, D)
    # 它返回 result, probs (或者只有 result，视实现而定)
    # test_attention_activation 中它返回了 result, probs
    
    q_ndl = ndl.Tensor(q_np, device=device)
    k_ndl = ndl.Tensor(k_np, device=device)
    v_ndl = ndl.Tensor(v_np, device=device)
    
    layer = nn.FlashMutiHeadAttention(
        dropout=dropout, causal=causal, device=device, dtype="float16", kernel=kernel)
        
    result_ndl = layer(q_ndl, k_ndl, v_ndl) # 这里假设输入 Q=K=V 测试 self-attention
    # 注意：上面的原始测试中只传了 q, q, q。为了通过测试我们最好也只传 q_ndl
    # 但为了更通用的对比，我们可以稍微修改成传入 distinct Q, K, V
    
    # 按照原始 test_attention_activation 逻辑重写：
    # result_ndl, probs_ndl = layer(q_ndl, k_ndl, v_ndl)
    
    res_ndl_val = result_ndl.numpy()
    
    # --- PyTorch 运行 ---
    # PyTorch 的 scaled_dot_product_attention 默认期望 (Batch, Num_Heads, Seq_Len, Head_Dim)
    # 正好也是 (B, H, L, D)
    
    q_torch = torch.from_numpy(q_np).to("cuda")
    k_torch = torch.from_numpy(k_np).to("cuda")
    v_torch = torch.from_numpy(v_np).to("cuda")
    
    out_torch = torch.nn.functional.scaled_dot_product_attention(
        q_torch, k_torch, v_torch,
        attn_mask=None,
        dropout_p=dropout,
        is_causal=causal
    )
    res_torch_val = out_torch.cpu().numpy()
    
    # --- 对比 ---
    np.testing.assert_allclose(res_ndl_val, res_torch_val, atol=1e-2, rtol=1e-2)