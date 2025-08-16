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

        self.dim = dim  # embedding_size of model
        self.hidden_dim = hidden_dim    # intermediate_size of MLP
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
        self.input_layernorm = RMSNorm(hidden_size, w_input_layernorm, rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, w_post_attention_layernorm, rms_norm_eps)

        self.multi_head_att = Qwen2MultiHeadAttention(hidden_size, num_attention_heads, num_kv_heads,
                                                      wq, wk, wv, wo, bq, bk, bv, max_seq_len, theta)
        self.mlp = Qwen2MLP(hidden_size, intermediate_size, w_gate, w_up, w_down)

    def __call__(
        self,
        x: mx.array,
        offset: int,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        out1 = self.multi_head_att(self.input_layernorm(x),
                                  offset=offset, mask=mask)
        out2 = out1 + x
        out3 = self.mlp(self.post_attention_layernorm(out2))
        out = out3 + out2
        return out
        

class Qwen2ModelWeek1:
    def __init__(self, mlx_model: Any, precision=mx.float16):
        '''ModelArgs(
            model_type='qwen2', hidden_size=3584, num_hidden_layers=28, 
            intermediate_size=18944, num_attention_heads=28, rms_norm_eps=1e-06,
            vocab_size=152064, num_key_value_heads=4, max_position_embeddings=32768,
            rope_theta=1000000.0, rope_traditional=False, rope_scaling=None,
            tie_word_embeddings=False
        )'''
        self.precision = precision
        self.hidden_size = mlx_model.args.hidden_size
        self.vocab_size = mlx_model.args.vocab_size
        self.rms_norm_eps = mlx_model.args.rms_norm_eps

        self.embedding = Embedding(
            self.vocab_size,
            self.hidden_size,
            dequantize_linear(mlx_model.model.embed_tokens).astype(self.precision)
        )

        def to_precision(d: dict) -> dict:
            return {k: v.astype(self.precision) for k, v in d.items()}

        self.layers = []
        for i in range(mlx_model.args.num_hidden_layers):
            mlp_w = to_precision({
                'w_gate': dequantize_linear(mlx_model.model.layers[i].mlp.gate_proj),
                'w_up': dequantize_linear(mlx_model.model.layers[i].mlp.up_proj),
                'w_down': dequantize_linear(mlx_model.model.layers[i].mlp.down_proj),
            })

            layernorm_w = to_precision({
                'w_input_layernorm': mlx_model.model.layers[i].input_layernorm.weight,
                'w_post_attention_layernorm': mlx_model.model.layers[i].post_attention_layernorm.weight,
            })

            self_attn_w = to_precision({
                'wq': dequantize_linear(mlx_model.model.layers[i].self_attn.q_proj),
                'wk': dequantize_linear(mlx_model.model.layers[i].self_attn.k_proj),
                'wv': dequantize_linear(mlx_model.model.layers[i].self_attn.v_proj),
                'wo': dequantize_linear(mlx_model.model.layers[i].self_attn.o_proj),
                'bq': mlx_model.model.layers[i].self_attn.q_proj.bias,
                'bk': mlx_model.model.layers[i].self_attn.k_proj.bias,
                'bv': mlx_model.model.layers[i].self_attn.v_proj.bias,
            })

            layer = Qwen2TransformerBlock(
                num_attention_heads=mlx_model.args.num_attention_heads,
                num_kv_heads=mlx_model.args.num_key_value_heads,
                hidden_size=self.hidden_size,
                intermediate_size=mlx_model.args.intermediate_size,
                rms_norm_eps=self.rms_norm_eps,
                **mlp_w,
                **layernorm_w,
                **self_attn_w,
                max_seq_len=mlx_model.args.max_position_embeddings,
                theta=mlx_model.args.rope_theta
            )
            self.layers.append(layer)

        self.norm = RMSNorm(
            dim=self.hidden_size,
            weight=mlx_model.model.norm.weight.astype(self.precision),
            eps=self.rms_norm_eps
        )
        
        # You can decide which strategy to use based on the mlx_model.args.tie_word_embeddings argument.
        # If it is true, then you should use Embedding::as_linear.
        # Otherwise, the lm_head linear layer will be available and you should load its parameters.
        if mlx_model.args.tie_word_embeddings:
            self.lm_head_w = None
        else:
            self.lm_head_w = dequantize_linear(mlx_model.lm_head)

    def __call__(
        self,
        inputs: mx.array,
        offset: int,
    ) -> mx.array:
        '''
        input
        | (tokens: N..)
        Embedding
        | (N.. x hidden_size); note that hidden_size==embedding_dim
        Qwen2TransformerBlock
        | (N.. x hidden_size)
        Qwen2TransformerBlock
        | (N.. x hidden_size)
        ...
        |
        RMSNorm 
        | (N.. x hidden_size)
        Embedding::as_linear  OR  Linear (lm_head)
        | (N.. x vocab_size)
        output
        '''
        out = self.embedding(inputs)
        for layer in self.layers:
            # set mask=causal when the input sequence is longer than 1.
            out = layer(out, offset, mask="causal" if out.shape[1] > 1 else None)
        out = self.norm(out)
        if self.lm_head_w is not None:
            return linear(out, self.lm_head_w)
        else:
            return self.embedding.as_linear(out)