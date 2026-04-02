#include "mul_sigmoid_nvidia.cuh"

#include "../../../utils.hpp"
#include "../../../utils/gpu_utils.hpp"

namespace {

template <typename T>
__global__ void mul_sigmoid_kernel(T *out, const T *in, const T *gate, size_t total) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + static_cast<size_t>(threadIdx.x);
    if (idx >= total) {
        return;
    }
    const float x = to_float(in[idx]);
    const float g = to_float(gate[idx]);
    out[idx] = from_float<T>(x / (1.0f + expf(-g)));
}

template <typename T>
void launch_mul_sigmoid(std::byte *out, const std::byte *in, const std::byte *gate, size_t total) {
    constexpr int block_size = 256;
    const int grid_size = static_cast<int>(CEIL(total, static_cast<size_t>(block_size)));
    mul_sigmoid_kernel<T><<<grid_size, block_size>>>(reinterpret_cast<T *>(out), reinterpret_cast<const T *>(in), reinterpret_cast<const T *>(gate), total);
}

} // namespace

namespace wginfer::ops::nvidia {

void mul_sigmoid(std::byte *out, const std::byte *in, const std::byte *gate, wginferDataType_t dtype, size_t M, size_t N) {
    const size_t total = M * N;
    if (total == 0) {
        return;
    }
    switch (dtype) {
    case WGINFER_DTYPE_F32:
        launch_mul_sigmoid<float>(out, in, gate, total);
        break;
    case WGINFER_DTYPE_F16:
        launch_mul_sigmoid<half>(out, in, gate, total);
        break;
    case WGINFER_DTYPE_BF16:
        launch_mul_sigmoid<__nv_bfloat16>(out, in, gate, total);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace wginfer::ops::nvidia
