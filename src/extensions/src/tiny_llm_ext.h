#pragma once

# include "mlx/utils.h"
# include "mlx/primitives.h"

namespace mx = mlx::core;

namespace tiny_llm_ext {
    /**
     * @brief Load a shared library for device-specific operations.
     *
     * This function dynamically loads a custom library (e.g., CUDA kernel)
     * for executing quantized operations on the given device.
     *
     * @param d      The device to load the library for (CPU or GPU).
     * @param path   The file path to the shared library (.so file).
     */
    void load_library(mx::Device d, const char *path);


    /**
     * @brief Perform a quantized matrix multiplication.
     *
     * This function provides a high-level API for quantized matmul.
     * It handles input validation and dispatches the actual computation
     * to the appropriate backend (CPU/GPU) via QuantizedMatmul primitive.
     *
     * @param scales        The per-group scaling factors.
     * @param biases        The per-group bias terms.
     * @param a             The input activation tensor.
     * @param b             The quantized weight tensor.
     * @param transpose_b   Whether to transpose matrix B before multiplication.
     * @param s             The execution stream or device context.
     *
     * @return The output tensor after quantized matrix multiplication.
     */
    mx::array quantized_matmul(
        const mx::array &scales,
        const mx::array &biases,
        const int group_size,
        const int bits,
        const mx::array &a, const mx::array &b,
        const bool transpose_b, mx::StreamOrDevice s
    );


    /**
     * @class QuantizedMatmul
     * @brief A low-level MLX primitive for quantized matrix multiplication.
     *
     * This class defines the core computation logic for quantized matmul.
     * It inherits from `mx::Primitive`, which represents a generic
     * computation node in the MLX framework.
     *
     * The class implements both CPU and GPU evaluation paths.
     */
    class QuantizedMatmul : public mx::Primitive {
        public:
            /**
            * @brief Construct a new QuantizedMatmul primitive.
            *
            * @param stream       The execution stream (CPU or GPU).
            * @param group_size   The number of elements per quantization group.
            * @param bits         The quantization bit-width (e.g., 4-bit).
            */
            explicit QuantizedMatmul(mx::Stream stream, const int group_size, const int bits)
                : mx::Primitive(stream), group_size_(group_size), bits_(bits) {};
        
            /**
            * @brief Execute quantized matmul on the CPU/GPU.
            *
            * @param inputs   A vector of input tensors (scales, biases, A, B).
            * @param outputs  A vector to store the resulting output tensor.
            */
            void eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override;
            void eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override;
        
            /**
            * @brief Vectorization mapping (not implemented for this primitive).
            *
            * Throws an exception since vmap is not supported for QuantizedMatmul.
            */
            std::pair<std::vector<mx::array>, std::vector<int>> vmap(const std::vector<mx::array> &inputs,
                                                                    const std::vector<int> &axes) override {
                throw std::runtime_error("QuantizedMatmul has no vmap implementation.");
            }
        
            /** @brief Get the name of the primitive. */
            const char *name() const override { return "QuantizedMatmul"; }
        
        private:
            int group_size_;    ///< Number of elements per quantization group
            int bits_;          ///< Number of bits for quantization, only 4-bit is supported
    };

}  // namespace tiny_llm_ext
