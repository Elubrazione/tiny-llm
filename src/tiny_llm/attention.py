import mlx.core as mx
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    d_k = query.shape[-1]   # D
    scale = 1 / mx.sqrt(mx.array(d_k, dtype=query.dtype)) if scale is None else scale
    # (N, H, L, D) @ (N, H, D, L) -> (N, H, L, L)
    scores = mx.matmul(query, key.transpose(*range(key.ndim - 2), -1, -2)) * scale
    if mask is not None:
        scores = scores + mask
    weights = mx.softmax(scores, -1)
    # (N x H x L x L) @ (N, H, L, D) => (N, H, L, D)
    out = mx.matmul(weights, value)
    return out

class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        # output/input: N x L x E
        # multi-heads input: N x L x H x D => N x H x L x D
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0
        self.head_dim = hidden_size // num_heads
        self.scale = mx.rsqrt(self.head_dim)
        
        # w_q/w_k/w_v: E x (H x D)
        assert wq.shape == (hidden_size, num_heads * self.head_dim)
        assert wk.shape == (hidden_size, num_heads * self.head_dim)
        assert wv.shape == (hidden_size, num_heads * self.head_dim)
        assert wo.shape == (num_heads * self.head_dim, hidden_size)

        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo    # w_o: (H x D) x E

    def _get_weighted_matrix(
        self,
        n_dim: int,
        l_dim: int,
        x: mx.array,
        w: mx.array
    ) -> mx.array:
        # (N x L x E) @ (E x E) => (N x L x E) => (N x L x H x D) => (N x H x L x D)
        return linear(x, w). \
            reshape(n_dim, l_dim, self.num_heads, self.head_dim). \
            transpose(0, 2, 1, 3)
    
    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        n_dim, l_dim, _ = query.shape
        weighted_q, weighted_k, weighted_v = [
            res for res in [
                self._get_weighted_matrix(n_dim, l_dim, in_x, in_w)
                for in_x, in_w in [(query, self.wq), (key, self.wk), (value, self.wv)]
            ]
        ]
        x = scaled_dot_product_attention_simple(weighted_q, weighted_k, weighted_v,
                                                scale=self.scale, mask=mask)
        x = x.transpose(0, 2, 1, 3).reshape(n_dim, l_dim, self.hidden_size)
        return linear(x, self.wo)
        


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    masked_matrix = mx.tril(mx.ones((L, S)), k = (S - L))
    causal_mask = mx.where(masked_matrix, mx.array(0), mx.array(-mx.inf)).astype(dtype)
    return causal_mask

def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    '''
    query: N.. x H_q x L x D
    key: N.. x H x S x D
    value: N.. x H x S x D
    mask: N.. x H_q x L x S
    output: N.. x H_q x L x D
    '''
    original_reshap = query.shape
    batch_dim = query.shape[: -3]
    q_head_num, l_dim, head_dim = query.shape[-3: ]
    kv_head_num, seq_dim, _ = key.shape[-3: ]
    assert q_head_num % kv_head_num == 0, \
        f"the number of query heads nust be divisible by the number of key/value heads!"
    n_repeats = q_head_num // kv_head_num
    
    # N.. is zero or more dimensions for batches
    query = query.reshape(*batch_dim, -1, kv_head_num, n_repeats, l_dim, head_dim)
    key = key.reshape(*batch_dim, -1, kv_head_num, 1, seq_dim, head_dim)
    value = value.reshape(*batch_dim, -1, kv_head_num, 1, seq_dim, head_dim)

    if mask is not None:
        if mask == "causal":
            mask = causal_mask(l_dim, seq_dim, query.dtype)
        # `mask` may not have the dimension of batch_dim and heads,
        # that means all batches and heads share the same mask, so we need to broadcast first
        mask = mx.broadcast_to(mask, (*batch_dim, q_head_num, l_dim, seq_dim))
        mask = mask.reshape(*batch_dim, 1, kv_head_num, n_repeats, l_dim, seq_dim)

    return scaled_dot_product_attention_simple(query, key, value, scale, mask).reshape(original_reshap)
    


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
) -> mx.array:
    pass
