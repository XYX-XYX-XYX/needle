from ..autograd import TensorTupleOp
from ..backend_selection import array_api, BACKEND


class FlashAttention(TensorTupleOp):
    def __init__(self, dropout, causal):
        self.dropout = dropout
        self.causal = causal

    def compute(self, q, k, v):
        if BACKEND != "nd":
            raise RuntimeError("FlashAttention requires the needle ndarray backend")

        if q.device != k.device or q.device != v.device:
            raise ValueError("FlashAttention expects q, k, and v to live on the same device")

        if getattr(q.device, "name", None) != "cuda":
            raise RuntimeError("FlashAttention only supports the CUDA backend")

        if len(q.shape) != 4 or len(k.shape) != 4 or len(v.shape) != 4:
            raise ValueError("FlashAttention expects q, k, and v with shape (batch_size, num_heads, seq_len, head_dim)")

        batch_size, num_heads, q_len, q_dim = q.shape
        k_batch_size, k_num_heads, kv_len, k_dim = k.shape
        v_batch_size, v_num_heads, v_len, v_dim = v.shape

        if batch_size != k_batch_size or batch_size != v_batch_size:
            raise ValueError("FlashAttention expects q, k, and v to share the same batch size")
        if num_heads != k_num_heads or num_heads != v_num_heads:
            raise ValueError("FlashAttention expects q, k, and v to share the same number of heads")
        if kv_len != v_len:
            raise ValueError("FlashAttention expects k and v to share the same sequence length")
        if q_dim != k_dim or q_dim != v_dim:
            raise ValueError("FlashAttention expects q, k, and v to share the same head dimension")

        if not getattr(q.device, "__has_flash_attention_stub__", False) or not hasattr(q.device, "flash_attention_forward"):
            raise RuntimeError(
                "FlashAttention backend hook is not registered; rebuild the CUDA backend with NEEDLE_USE_FLASHATTN_STUB=ON and CUTLASS available"
            )

        q = q.compact()
        k = k.compact()
        v = v.compact()

        result = array_api.empty(q.shape, device=q.device)
        probs = array_api.empty((batch_size, num_heads, q_len, kv_len), device=q.device)

        q.device.flash_attention_forward(
            q._handle,
            k._handle,
            v._handle,
            result._handle,
            probs._handle,
            batch_size,
            num_heads,
            q_len,
            kv_len,
            q_dim,
            self.dropout,
            self.causal,
        )
        return result, probs

    def gradient(self, out_grad, node):
        raise NotImplementedError("FlashAttention forward hook is registered, but backward is not implemented")


def flashattention(q, k, v, dropout, causal):
    return FlashAttention(dropout, causal)(q, k, v)
