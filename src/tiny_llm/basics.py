import mlx.core as mx
import math


def softmax(x: mx.array, axis: int) -> mx.array:
    # TODO: manual implementation
    return mx.softmax(x, axis=axis)


def linear(
    x: mx.array,
    w: mx.array,
    bias: mx.array | None = None,
) -> mx.array:
    out = mx.matmul(x, w.T)
    if bias is not None:
        # (O, ) will be auto-broadcasted into (N, O)
        out = out + bias
    return out

def silu(x: mx.array) -> mx.array:
    '''
    Takes a tensor of the shape (N.. x I) and returns a tensor of the same shape.
    SiLU(x) = x * sigmoid(x) = \frac{x}{1 + e^{-x}}
    '''
    return x / (1 + mx.exp(-x))
