#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include "wginfer.h"
#include <cmath>
#include <cstddef>

template<typename T>
void rms_norm_(T *out, const T *in, const T *weight, size_t M, size_t N, float eps) {
    for (size_t m = 0; m < M; m++) {
        // 1. 计算当前行的均方
        float sum = 0.0f;
        for (size_t n = 0; n < N; n++) {
            float value = wginfer::utils::cast<float>(in[m * N + n]);
            sum += value * value;
        }
        float mean = sum / static_cast<float>(N);
        float scale_rms = 1.0f / std::sqrt(mean + eps);

        // 2. 乘以权重并归一化
        for (size_t n = 0; n < N; n++) {
            float value = wginfer::utils::cast<float>(in[m * N + n]);
            float wei = wginfer::utils::cast<float>(weight[n]);
            float res = value * wei * scale_rms;
            out[m * N + n] = wginfer::utils::cast<T>(res);
        }
    }
}

namespace wginfer::ops::cpu {
void rms_norm(std::byte *out,
               const std::byte *in,
               const std::byte *weight,
               wginferDataType_t dataType,
               size_t M, size_t N, float eps){
    switch (dataType) {
    case WGINFER_DTYPE_F16:
        return rms_norm_(reinterpret_cast<wginfer::fp16_t *>(out), 
                         reinterpret_cast<const wginfer::fp16_t *>(in), 
                         reinterpret_cast<const wginfer::fp16_t *>(weight), 
                         M, N, eps);
    case WGINFER_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<wginfer::bf16_t *>(out), 
                         reinterpret_cast<const wginfer::bf16_t *>(in), 
                         reinterpret_cast<const wginfer::bf16_t *>(weight), 
                         M, N, eps);
    case WGINFER_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out), 
                         reinterpret_cast<const float *>(in), 
                         reinterpret_cast<const float *>(weight), 
                         M, N, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dataType);
    }
}
} // namespace wginfer::ops::cpu