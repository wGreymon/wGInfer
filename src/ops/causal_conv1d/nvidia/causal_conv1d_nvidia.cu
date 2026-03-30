#include "causal_conv1d_nvidia.cuh"

#include "../../../utils.hpp"
#include "../../../utils/gpu_utils.hpp"

#include <cmath>

namespace {

template <typename T>
__global__ void causal_conv1d_kernel(
    T *out,
    const T *in,
    const T *weight,
    size_t seqlen,
    size_t channels,
    size_t kernel_size) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = seqlen * channels;
    if (idx >= total) {
        return;
    }

    const size_t seq = idx / channels;
    const size_t channel = idx % channels;
    float acc = 0.0f;
    for (size_t k = 0; k < kernel_size; ++k) {
        const size_t input_seq = seq + k;
        if (input_seq >= kernel_size - 1 && (input_seq - (kernel_size - 1)) < seqlen) {
            const size_t src_seq = input_seq - (kernel_size - 1);
            const float x = to_float(in[src_seq * channels + channel]);
            const float w = to_float(weight[channel * kernel_size + k]);
            acc += x * w;
        }
    }
    const float silu = acc / (1.0f + expf(-acc));
    out[idx] = from_float<T>(silu);
}

template <typename T>
void launch_causal_conv1d(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    size_t seqlen,
    size_t channels,
    size_t kernel_size) {
    constexpr int block_size = 256;
    const size_t total = seqlen * channels;
    const int grid_size = static_cast<int>(CEIL(total, static_cast<size_t>(block_size)));
    causal_conv1d_kernel<T><<<grid_size, block_size>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(in),
        reinterpret_cast<const T *>(weight),
        seqlen,
        channels,
        kernel_size);
}

} // namespace

namespace wginfer::ops::nvidia {

void causal_conv1d(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    wginferDataType_t dtype,
    size_t seqlen,
    size_t channels,
    size_t kernel_size) {
    if (seqlen == 0 || channels == 0 || kernel_size == 0) {
        return;
    }

    switch (dtype) {
    case WGINFER_DTYPE_F32:
        launch_causal_conv1d<float>(out, in, weight, seqlen, channels, kernel_size);
        break;
    case WGINFER_DTYPE_F16:
        launch_causal_conv1d<half>(out, in, weight, seqlen, channels, kernel_size);
        break;
    case WGINFER_DTYPE_BF16:
        launch_causal_conv1d<__nv_bfloat16>(out, in, weight, seqlen, channels, kernel_size);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }

    CUDA_CHECK(cudaGetLastError());
}

} // namespace wginfer::ops::nvidia
