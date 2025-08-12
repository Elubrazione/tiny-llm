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
        Multi-head attention forward pass with RoPE (Rotary Positional Embedding)
        and grouped attention.

        Args:
            x: Input tensor of shape (B, L, E)
                - B = batch size
                - L = sequence length
                - E = embedding dimension (hidden size)
            offset: Starting position index for RoPE (used for streaming or segmented input)
            mask: Optional attention mask (tensor or special string type)
        Returns:
            Output tensor of shape (B, L, E)
        '''
        B, L, E = x.shape
        assert E == self.hidden_size, "Expected E == self.hidden_size!"
        
        # Project input x into Query, Key, and Value tensors, shape after projection:
        # query => (B, L, H_q, D), key => (B, L, H, D), value => (B, L, H, D)
        query, key, value = tuple(map(
            lambda x: self._get_weighted_matrix_without_transpose(*x),
            zip(
                [x] * 3,                          # same input for Q, K, V
                [self.wq, self.wk, self.wv],      # corresponding weights
                [self.bq, self.bk, self.bv],      # corresponding biases
                [self.num_heads] + [self.num_kv_heads] * 2  # head counts for Q and KV
            )
        ))

        # Apply rotary RoPE to query and key
        query, key = tuple(map(
            lambda it: self.rope(it, offset=slice(offset, offset + L)), [query, key]
        ))

        # Perform scaled dot-product attention, swap axes (B, L, H, D) => (B, H, L, D)
        # After attention swap back and reshape to (B, H_q, L, D) => (B, L, H_q, D) => (B, L, E)
        out = scaled_dot_product_attention_grouped(
            query.swapaxes(-2, -3), key.swapaxes(-2, -3), value.swapaxes(-2, -3), mask=mask
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
        assert w_gate.shape == (hidden_dim, dim)
        assert w_up.shape == (hidden_dim, dim)
        assert w_down.shape == (dim, hidden_dim)

        self.dim = dim  # intermediate_size
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate    # (I x E)
        self.w_up = w_up        # (I x E)
        self.w_down = w_down    # (E x I)

    def __call__(self, x: mx.array) -> mx.array:
        out = silu(linear(x, self.w_gate)) * linear(x, self.w_up)
        return linear(out, self.w_down)


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
