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
        
        # w_i = base ^ {-(2 * i) / half_dim}} for i = 0 to half_dim, where i is the index of embedding length.
        # since every pair in single token embedding share the same rotated frequency, here the length is equal to embedding length // 2.
        index = mx.arange(0, self.half_dim, dtype=mx.float32) / self.half_dim   # (D // 2, )
        wi = mx.power(base, -index) # (D // 2, )
        # outer((MAX_SEQ_LEN, ), (D // 2, )) => (MAX_SEQ_LEN, D // 2)
        freqs = mx.outer(mx.arange(seq_len), wi)
        
        # cos/sin_freqs: (MAX_SEQ_LEN, D // 2)
        # merge_freqs: (MAX_SEQ_LEN, D // 2, 2)
        self.freqs = mx.stack([mx.cos(freqs), mx.sin(freqs)], axis=-1)

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
        
        re_x = x.reshape(n_dim, l_dim, head_num, head_dim // 2, 2)
        re_x0 = re_x[..., 0]
        re_x1 = re_x[..., 1]

        # merge_freqs: (L, D // 2, 2) => (1 or N, L, 1, D // 2, 2)
        # it will be broadcosted into (N, L, H, D // 2, 1) when @ with x (N, L, H, D // 2, 1)
        cos = freqs[..., 0]
        sin = freqs[..., 1]

        out0 = re_x0 * cos - re_x1 * sin
        out1 = re_x1 * cos + re_x0 * sin
        out = mx.stack([out0, out1], axis=-1)  # (N, L, H, D // 2, 2)

        return out.reshape(n_dim, l_dim, head_num, head_dim).astype(x.dtype)    # (N, L, D)


            
