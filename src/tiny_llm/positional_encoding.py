import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        assert dims % 2 == 0
        self.dims = dims
        self.half_dim = dims // 2
        self.seq_len = seq_len
        self.base = base
        self.traditional = traditional
        
        # w_i = base ^ {-(2 * i) / dims}} = base ^ {-i / self.half_dim} for i = 0 to half_dim, where i is the index of embedding length.
        # since every pair in single token embedding share the same rotated frequency, here the length is equal to embedding length // 2.
        index = mx.arange(0, self.half_dim, dtype=mx.float32) / self.half_dim   # (D // 2, )
        wi = mx.power(base, -index) # (D // 2, )
        # outer((MAX_SEQ_LEN, ), (D // 2, )) => (MAX_SEQ_LEN, D // 2)
        freqs = mx.outer(mx.arange(seq_len), wi)
        
        # cos/sin_freqs: (MAX_SEQ_LEN, D // 2)
        # merge_freqs: (MAX_SEQ_LEN, D // 2, 2)
        self.freqs = mx.stack([mx.cos(freqs), mx.sin(freqs)], axis=-1)
        # self.freqs 的形状将是 (seq_len, D // 2, 2)，每个位置的频率对 (freq_0, freq_1, ..., freq_{D//2-1}) 
        # 将被扩展为 (cos(freq_0), sin(freq_0)), (cos(freq_1), sin(freq_1)), ...

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        # x: (N, L, H, D) => (N, L, H, D // 2, 2)
        n_dim, l_dim, head_num, head_dim = x.shape
        assert head_dim // 2 == self.half_dim, f"Expected dim {self.dims}, got {head_dim}"
        if offset is not None and isinstance(offset, list):
            offset = mx.array([list(range(s.start, s.stop)) for s in offset])  # (N, L)

        freqs = self.freqs[: l_dim, :, :] if offset is None else self.freqs[offset, :, :]
        freqs = freqs.reshape(-1, l_dim, 1, self.half_dim, 2)  # add head dim
        # 广播使得每个头部都可以共享相同的 (D // 2, 2) 的频率对（余弦和正弦）
        
        re_x = x.reshape(n_dim, l_dim, head_num, head_dim // 2, 2)
        if self.traditional:
            '''
            output[0] = x[0] * cos_freqs[0] + x[1] * -sin_freqs[0]
            output[1] = x[0] * sin_freqs[0] + x[1] * cos_freqs[0]
            output[2] = x[2] * cos_freqs[1] + x[3] * -sin_freqs[1]
            output[3] = x[2] * sin_freqs[1] + x[3] * cos_freqs[1]
            '''
            re_x0 = re_x[..., 0]
            re_x1 = re_x[..., 1]
        else:
            '''
            The Qwen2 model uses a non-traditional form of RoPE. 
            In this form, the head embedding dimension is split into two halves, 
                and the two halves are applied with different frequencies. 
            Let's say x1 = x[.., :HALF_DIM] and x2 = x[.., HALF_DIM:].
            
            output[0] = x1[0] * cos_freqs[0] + x2[0] * -sin_freqs[0]
            output[HALF_DIM] = x1[0] * sin_freqs[0] + x2[0] * cos_freqs[0]
            output[1] = x1[1] * cos_freqs[1] + x2[1] * -sin_freqs[1]
            output[HALF_DIM + 1] = x1[1] * sin_freqs[1] + x2[1] * cos_freqs[1]
            '''
            re_x0 = x[..., 0: self.half_dim]
            re_x1 = x[..., self.half_dim:]

        # merge_freqs: (L, D // 2, 2) => (1 or N, L, 1, D // 2, 2)
        # it will be broadcosted into (N, L, H, D // 2, 2) when @ with x (N, L, H, D // 2, 2)
        cos = freqs[..., 0] # (N, L, H, D // 2)
        sin = freqs[..., 1] # (N, L, H, D // 2)

        out0 = re_x0 * cos - re_x1 * sin    # (N, L, H, D // 2)
        out1 = re_x1 * cos + re_x0 * sin    # (N, L, H, D // 2)
        out = (mx.stack if self.traditional else mx.concat)([out0, out1], axis=-1)  # (N, L, H, D // 2, 2) or (N, L, H, D) 
        return out.reshape(n_dim, l_dim, head_num, head_dim)    # (N, L, H, D)


            
