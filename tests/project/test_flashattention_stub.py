import sys
import re

sys.path.append("./python")

import numpy as np
import pytest

import needle as ndl
from needle.backend_ndarray import ndarray as ndarray_api
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


def _tensor_or_skip(array, device):
    try:
        return ndl.Tensor(array, device=device)
    except RuntimeError as err:
        if "CUDA driver version is insufficient for CUDA runtime version" in str(err):
            pytest.skip("CUDA runtime allocation is unavailable in this environment")
        raise


@pytest.mark.parametrize("kernel", ["auto", "sm80", "sm90"])
@pytest.mark.parametrize("device", [CUDA_DEVICE])
def test_flashattention_module_accepts_kernel_selector(kernel, device):
    layer = nn.FlashMutiHeadAttention(dropout=0.0, causal=False, device=device, kernel=kernel)

    assert layer.kernel == kernel


class _FakeHandle:
    pass


class _FakeArray:
    def __init__(self, shape, device):
        self.shape = shape
        self.device = device
        self._handle = _FakeHandle()

    def compact(self):
        return self


class _FakeCudaDevice:
    name = "cuda"
    __has_flash_attention_stub__ = True

    def __init__(self):
        self.calls = []

    def empty(self, shape, dtype="float32"):
        assert dtype == "float32"
        return _FakeArray(shape, self)

    def flash_attention_forward(self, *args):
        self.calls.append(args)
        raise RuntimeError("flash attention kernel registered but not implemented")


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dropout", [0.0, 0.1])
@pytest.mark.parametrize("kernel", ["auto", "sm80", "sm90"])
def test_flashattention_stub_calls_cuda_hook(causal, dropout, kernel):
    device = _FakeCudaDevice()
    q = _FakeArray((2, 4, 8, 16), device)
    k = _FakeArray((2, 4, 8, 16), device)
    v = _FakeArray((2, 4, 8, 16), device)

    with pytest.raises(RuntimeError, match="registered but not implemented"):
        ndarray_api.flash_attention(q, k, v, dropout, causal, kernel)

    assert len(device.calls) == 1
    hook_args = device.calls[0]
    assert hook_args[0] is q._handle
    assert hook_args[1] is k._handle
    assert hook_args[2] is v._handle
    assert hook_args[3] is not None
    assert hook_args[4:] == (2, 4, 8, 8, 16, dropout, causal, kernel)


def test_flashattention_rejects_unknown_kernel():
    device = _FakeCudaDevice()
    q = _FakeArray((2, 4, 8, 16), device)
    k = _FakeArray((2, 4, 8, 16), device)
    v = _FakeArray((2, 4, 8, 16), device)

    with pytest.raises(ValueError, match="kernel"):
        ndarray_api.flash_attention(q, k, v, 0.0, False, "bad")
