from ..autograd import TensorOp
from ..backend_selection import array_api, BACKEND


class FlashAttention(TensorOp):
    def __init__(self, dropout, causal):
        self.dropout = dropout
        self.causal = causal

    def compute(self, q, k, v):
        if BACKEND != "nd":
            raise RuntimeError("FlashAttention requires the needle ndarray backend")
        return array_api.flash_attention(q, k, v, self.dropout, self.causal)

    def gradient(self, out_grad, node):
        raise NotImplementedError("FlashAttention forward hook is registered, but backward is not implemented")


def flashattention(q, k, v, dropout, causal):
    return FlashAttention(dropout, causal)(q, k, v)
