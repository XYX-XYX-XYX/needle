from needle import ops
from .nn_basic import Module

class FlashMutiHeadAttention(Module):

    def __init__(
        self,
        *,
        dropout = 0.,
        causal = False,
        device = None,
        dtype = "float32",
    ):
        device_name = getattr(device, "name", device)
        if device_name != "cuda":
            raise ValueError("flash attention only supports cuda")
        super().__init__()

        self.dropout = dropout
        self.causal = causal
        self.device = device
        self.dtype = dtype

    def forward(
        self,
        q, k, v
    ):
        batch_size, num_head, queries_len, q_dim = q.shape
        _, _, keys_values_len, k_dim = k.shape
        _, _, _, v_dim = v.shape

        assert q_dim == k_dim == v_dim

        return ops.flashattention(q, k, v, dropout=self.dropout, causal=self.causal)
