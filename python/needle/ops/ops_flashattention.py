from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *

class FlashAttention(TensorOp):
    def __init__(self, dropout, causal):
        self.dropout = dropout
        self.causal = causal

    def compute(self, q, k, v):
        raise NotImplementedError("FlashAttention backend not implemented")
    
    def gradient(self, out_grad, node):
        raise NotImplementedError("FlashAttention grad not implemented!")
    
def flashattention(q, k, v, dropout, causal):
    return FlashAttention(dropout, causal)(q, k, v)

    