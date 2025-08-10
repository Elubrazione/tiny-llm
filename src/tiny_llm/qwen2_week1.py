import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        assert hidden_size % num_heads == 0
        self.head_dim = hidden_size // num_heads
        self.scale = mx.rsqrt(self.head_dim)

        for it in [wq, wo]:
            assert it.shape == (num_heads * self.head_dim, hidden_size)
        for it in [wk, wv]:
            assert it.shape == (num_kv_heads * self.head_dim, hidden_size)

        assert num_heads % num_kv_heads == 0
        self.n_repeats = num_heads // num_kv_heads
        
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.hidden_size = hidden_size
        
        self.wq, self.bq = wq, bq
        self.wk, self.bk = wk, bk
        self.wv, self.bv = wv, bv
        self.wo = wo
        
        # rope.traditional: False (default), same as in Qwen2
        self.rope = RoPE(dims=self.head_dim, seq_len=max_seq_len, base=theta)
        
    def _get_weighted_matrix_without_transpose(
        self,
        x: mx.array,
        w: mx.array,
        b: mx.array,
        h: int,
    ):
        n_dim, l_dim, _ = x.shape
        # (B x L x E) @ ((H_q/H x D) x E)^T => (B x L x (H_q/H x D))
        # => (B x L x (H_q/H x D)) => (B x L x H_q/H x D)
        return linear(x, w, b). \
            reshape(n_dim, l_dim, h, self.head_dim)

    def __call__(
        self,
        x: mx.array,
        offset: int,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        '''
        x: B, L, E
        q = linear(x, wq, bq) -> B, L, H_q, D
        k = linear(x, wk, bk) -> B, L, H, D
        v = linear(x, wv, bv) -> B, L, H, D
        q = rope(q, offset=slice(offset, offset + L))
        k = rope(k, offset=slice(offset, offset + L))
        (transpose as needed)
        x = scaled_dot_product_attention_grouped(q, k, v, scale, mask) -> B, L, H_q, D ; Do this at float32 precision
        (transpose as needed)
        x = linear(x, wo) -> B, L, E
        '''
        B, L, E = x.shape
        assert E == self.hidden_size, "Expected E == self.hidden_size!"
        query, key, value = tuple(map(
            lambda x: self._get_weighted_matrix_without_transpose(*x),
            zip(
                [x] * 3, [self.wq, self.wk, self.wv],
                [self.bq, self.bk, self.bv], [self.num_heads] + [self.num_kv_heads] * 2
            )
        ))
        query, key = tuple(map(lambda it: \
            self.rope(it, offset=slice(offset, offset + L)), [query, key]))
        
        # (B x H_q x L x D) => (B x L x H_q x D) => (B x L x E)
        out = scaled_dot_product_attention_grouped(
            query.swapaxes(-2, -3), key.swapaxes(-2, -3),
            value.swapaxes(-2, -3), mask=mask
        ).swapaxes(-2, -3).reshape(B, L, E)

        return linear(out, self.wo)
        
        
        
        
        


class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        pass

    def __call__(self, x: mx.array) -> mx.array:
        pass


class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        pass

    def __call__(
        self,
        x: mx.array,
        offset: int,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        pass


class Qwen2ModelWeek1:
    def __init__(self, mlx_model: Any):
        pass

    def __call__(
        self,
        inputs: mx.array,
        offset: int,
    ) -> mx.array:
        pass
