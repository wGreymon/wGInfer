#include "swiglu_cpu.hpp"
#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void swiglu_(T *out, const T *gate, const T *up, size_t numel) {
    for (size_t i = 0; i < numel; i++) {
        float gate_val = wginfer::utils::cast<float>(gate[i]);
        float up_val = wginfer::utils::cast<float>(up[i]);
        float res = up_val * gate_val / (1 + std::exp(-gate_val));
        out[i] = wginfer::utils::cast<T>(res);
    }
}

namespace wginfer::ops::cpu {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, 
            wginferDataType_t type, size_t numel) {
    switch (type) {
    case WGINFER_DTYPE_F16:
        return swiglu_(reinterpret_cast<fp16_t *>(out), reinterpret_cast<const fp16_t *>(gate), reinterpret_cast<const fp16_t *>(up), numel);
    case WGINFER_DTYPE_BF16:
        return swiglu_(reinterpret_cast<bf16_t *>(out), reinterpret_cast<const bf16_t *>(gate), reinterpret_cast<const bf16_t *>(up), numel);
    case WGINFER_DTYPE_F32:
        return swiglu_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(gate), reinterpret_cast<const float *>(up), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace wginfer::ops::cpu