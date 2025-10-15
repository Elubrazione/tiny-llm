# include <mlx/array.h>
# include "mlx/backend/common/utils.h"
# include "mlx/backend/cpu/encoder.h"
# include "tiny_llm_ext.h"

using namespace std;

namespace tiny_llm_ext {
    /*
        User calls quantized_matmul()
                │
                ▼
        Create mx::array node
        ├─ shape/dtype → knows output size & type
        ├─ primitive  → knows how to compute
        └─ inputs     → knows dependent tensors
                │
                ▼
        Build computation graph & allocate memory
                │
                ▼
        Call eval_cpu() / eval_gpu() to execute
                │
                ▼
        Results are stored in mx::array
    */
    mx::array quantized_matmul(
        const mx::array &scales, const mx::array &biases,
        const int group_size, const int bits,
        const mx::array &a, const mx::array &b,
        const bool transpose_b,
        mx::StreamOrDevice device    // cpu or gpu
    ) {
        auto out_shape = a.shape();
        out_shape[1] = b.shape()[0];
        return mx::array(
            out_shape, a.dtype(),
            make_shared<QuantizedMatmul>(to_stream(device), group_size, bits),
            {scales, biases, a, b}
        );
    }

    void quantized_matmul_impl(const mx::array &scales, const mx::array &biases, const mx::array &a, const mx::array &b,
                                mx::array &out, int group_size, int bits, mx::Stream stream
    ) {
        out.set_data(mx::allocator::malloc(out.nbytes()));
        auto &encoder = mx::cpu::get_command_encoder(stream);
        encoder.set_input_array(scales);
        encoder.set_input_array(biases);
        encoder.set_input_array(a);
        encoder.set_input_array(b);
        encoder.set_output_array(out);
        
        encoder.dispatch([out_ptr = out.data<float16_t>(), out_shape = out.shape(), out_strides = out.strides(),
                        a = mx::array::unsafe_weak_copy(a), b = mx::array::unsafe_weak_copy(b),
                        scales = mx::array::unsafe_weak_copy(scales), biases = mx::array::unsafe_weak_copy(biases)]() {
            int M = a.shape()[0];
            int N = a.shape()[1];
            int K = b.shape()[0];
            const int group_size = 64;
            const int bits = 4;
            const int group_per_row = N / group_size;
            const float16_t *a_ptr = a.data<float16_t>();
            const uint32_t *b_ptr = b.data<uint32_t>();
            const float16_t *scales_ptr = scales.data<float16_t>();
            const float16_t *biases_ptr = biases.data<float16_t>();
            uint32_t item_mask = (1 << bits) - 1;
            for (int i = 0; i < M; i++) {
                for (int k = 0; k < K; k++) {
                    float sum = 0;
                    for (int group_idx = 0; group_idx < group_per_row; group_idx++) {
                        int64_t scales_loc =
                            mx::elem_to_loc(k * group_per_row + group_idx, scales.shape(), scales.strides());
                        int64_t biases_loc =
                            mx::elem_to_loc(k * group_per_row + group_idx, biases.shape(), biases.strides());
                        float16_t scale = scales_ptr[scales_loc];
                        float16_t bias = biases_ptr[biases_loc];
                        int64_t b_loc = mx::elem_to_loc((k * N + group_idx * group_size) / 8, b.shape(), b.strides());
                        int64_t a_loc = mx::elem_to_loc(i * N + group_idx * group_size, a.shape(), a.strides());
                        const int packs_per_item = 32 / bits;
                        for (int item_idx = 0; item_idx < group_size; item_idx += packs_per_item) {
                            uint32_t b_val = b_ptr[b_loc];
                            uint8_t *b_bytes = reinterpret_cast<uint8_t *>(&b_val);
                            for (int pack_idx = 0; pack_idx < packs_per_item; pack_idx++) {
                                uint8_t item_val = (b_bytes[pack_idx / 2] >> ((pack_idx % 2) * bits)) & item_mask;
                                float b = static_cast<float>(item_val) * scale + bias;
                                float a = a_ptr[a_loc];
                                sum += a * b;
                                a_loc += 1;
                            }
                            b_loc += 1;
                        }
                    }
                    int64_t out_loc = mx::elem_to_loc(i * K + k, out_shape, out_strides);
                    out_ptr[out_loc] = static_cast<float16_t>(sum);
                }
            }
        });
    }

    void QuantizedMatmul::eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
        const mx::array &scales = inputs[0];
        const mx::array &biases = inputs[1];
        const mx::array &a = inputs[2];
        const mx::array &b = inputs[3];
        mx::array &out = outputs[0];

        quantized_matmul_impl(scales, biases, a, b, out, group_size_, bits_, stream());
    }

    void QuantizedMatmul::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
        const mx::array &scales = inputs[0];
        const mx::array &biases = inputs[1];
        const mx::array &a = inputs[2];
        const mx::array &b = inputs[3];
        mx::array &out = outputs[0];

        quantized_matmul_impl(scales, biases, a, b, out, group_size_, bits_, stream());
    }
}