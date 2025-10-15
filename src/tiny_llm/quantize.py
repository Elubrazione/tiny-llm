import mlx.core as mx
from typing import Any
from extensions import tiny_llm_ext


def dequantize_linear(mx_layer: Any) -> mx.array:
    w = mx.dequantize(
        mx_layer.weight,
        mx_layer.scales,
        mx_layer.biases,
        mx_layer.group_size,
        mx_layer.bits,
    )
    return w


class QuantizedWeights:
    def __init__(
        self,
        scales: mx.array,
        biases: mx.array,
        group_size: int,
        bits: int,
        weight: mx.array,
    ):
        self.scales = scales
        self.biases = biases
        self.group_size = group_size
        self.bits = bits
        self.weight = weight

    @staticmethod
    def from_mlx_layer(mlx_layer: Any) -> "QuantizedWeights":
        return QuantizedWeights(
            scales=mlx_layer.scales,
            biases=mlx_layer.biases,
            group_size=mlx_layer.group_size,
            bits=mlx_layer.bits,
            weight=mlx_layer.weight,
        )

def quantized_matmul_python_ver(
    scales: mx.array,
    biases: mx.array,
    group_size: int,
    bits: int,
    a: mx.array,
    b: mx.array,
    transpose_b: bool = False,
) -> mx.array:
    assert bits == 4, "Only 4-bit quantization is supported"

    *N, D = a.shape
    a_flat = a.reshape(-1, D)
    a_flat = mx.contiguous(a_flat)
    b = mx.contiguous(b)

    num_groups = b.shape[0]
    decoded_cols = []

    for g in range(num_groups):
        packed = b[g]  # shape (ceil(group_size * bits / 32),)
        
        # uint4 unpack
        cols = []
        for word in packed:
            for shift in range(0, 32, 4):  # each uint32 has 8 4-bit
                val = (word >> shift) & 0xF  # shift 4 bits to the right and mask with 0xF
                cols.append(val)
        cols = mx.array(cols[: group_size], dtype=mx.float32)
        cols = cols * float(scales[g].item()) + float(biases[g].item())
        decoded_cols.append(cols)

    w_real = mx.stack(decoded_cols, axis=1)  # shape (D, E)
    if not transpose_b:
        w_real = w_real.T  # shape (E, D)

    out = mx.matmul(a_flat.astype(mx.float32), w_real.astype(mx.float32))
    out = out.reshape(*N, -1)
    return out

def quantized_matmul(
    scales: mx.array,
    biases: mx.array,
    group_size: int,
    bits: int,
    a: mx.array,
    b: mx.array,
    transpose_b: bool = False,
) -> mx.array:
    *N, D = a.shape
    a = a.reshape(-1, D)
    a = mx.contiguous(a)
    b = mx.contiguous(b)
    return tiny_llm_ext.quantized_matmul(
        scales, biases, group_size, bits, a, b, transpose_b
    ).reshape(*N, -1)


def quantized_linear(
    x: mx.array,
    w: QuantizedWeights,
    bias: mx.array | None = None,
) -> mx.array:
    # result = quantized_matmul(w.scales, w.biases, w.group_size, w.bits, x, w.weight, True)
    result = quantized_matmul(w.scales, w.biases, w.group_size, w.bits, x, w.weight, True)
    if bias is not None:
        result = result + bias
    return result
