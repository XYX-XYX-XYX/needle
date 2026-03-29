import sys

sys.path.append("./python")

import numpy as np
import pytest

import needle as ndl
import needle.nn as nn


CUDA_DEVICE = pytest.param(
    ndl.cuda(),
    marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU"),
    id="cuda",
)


def test_flashattention_module_rejects_non_cuda_device():
    with pytest.raises(ValueError, match="cuda"):
        nn.FlashMutiHeadAttention(device=ndl.cpu())


@pytest.mark.parametrize("device", [CUDA_DEVICE])
def test_flashattention_backend_hook_registered(device):
    assert getattr(device, "__has_flash_attention_stub__", False)
    assert hasattr(device, "flash_attention_forward")


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dropout", [0.0, 0.1])
@pytest.mark.parametrize("device", [CUDA_DEVICE])
def test_flashattention_stub_raises_clear_error(causal, dropout, device):
    np.random.seed(19943)
    q = np.random.randn(2, 4, 8, 16).astype(np.float32)

    layer = nn.FlashMutiHeadAttention(dropout=dropout, causal=causal, device=device)

    with pytest.raises(RuntimeError, match="registered but not implemented"):
        layer(
            ndl.Tensor(q, device=device),
            ndl.Tensor(q, device=device),
            ndl.Tensor(q, device=device),
        )
