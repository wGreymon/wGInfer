#include "mul_sigmoid_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

namespace {

template <typename T>
void mul_sigmoid_impl(T *out, const T *in, const T *gate, size_t M, size_t N) {
    for (size_t i = 0; i < M * N; ++i) {
        const float x = wginfer::utils::cast<float>(in[i]);
        const float g = wginfer::utils::cast<float>(gate[i]);
        out[i] = wginfer::utils::cast<T>(x / (1.0f + std::exp(-g)));
    }
}

} // namespace

namespace wginfer::ops::cpu {

void mul_sigmoid(std::byte *out, const std::byte *in, const std::byte *gate, wginferDataType_t dtype, size_t M, size_t N) {
    switch (dtype) {
    case WGINFER_DTYPE_F32:
        return mul_sigmoid_impl(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(gate), M, N);
    case WGINFER_DTYPE_F16:
        return mul_sigmoid_impl(reinterpret_cast<wginfer::fp16_t *>(out), reinterpret_cast<const wginfer::fp16_t *>(in), reinterpret_cast<const wginfer::fp16_t *>(gate), M, N);
    case WGINFER_DTYPE_BF16:
        return mul_sigmoid_impl(reinterpret_cast<wginfer::bf16_t *>(out), reinterpret_cast<const wginfer::bf16_t *>(in), reinterpret_cast<const wginfer::bf16_t *>(gate), M, N);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace wginfer::ops::cpu
