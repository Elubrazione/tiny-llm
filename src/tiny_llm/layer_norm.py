import mlx.core as mx


class RMSNorm:
    '''
    RMSNorm (Root Mean Square Normalization)

    Parameters
    ----------
    D : int
        Size of the embedding dimension (last dimension of x).
    x : mx.array
        Input tensor of shape (N..., D), where N... represents any number of leading dimensions
        such as batch size or sequence length.
    weight : mx.array
        Learnable scale parameter of shape (D, ) applied elementwise to the normalized output.

    Returns
    -------
    mx.array
        Tensor of shape (N..., D) with RMS normalization applied along the last dimension.
    '''
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        self.dim = dim
        self.weight = weight
        self.eps = eps

        # (N.. x D) => (N.. x 1), will be broadcasted into (N.. x D) later
        self.mean_rsqrt = lambda x: mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + eps)

    def __call__(self, x: mx.array) -> mx.array:
        x = x.astype(mx.float32)
        return x * self.mean_rsqrt(x) * self.weight
