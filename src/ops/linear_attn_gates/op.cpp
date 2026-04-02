#include "op.hpp"

#include "../../core/wginfer_core.hpp"
#include "../../utils.hpp"

#include "./cpu/linear_attn_gates_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "./nvidia/linear_attn_gates_nvidia.cuh"
#endif

namespace wginfer::ops {

void linear_attn_gates(tensor_t out_g, tensor_t out_beta, tensor_t a, tensor_t b, tensor_t a_log, tensor_t dt_bias) {
    CHECK_SAME_DEVICE(out_g, out_beta, a, b, a_log, dt_bias);
    CHECK_SAME_DTYPE(out_g->dtype(), out_beta->dtype(), a->dtype(), b->dtype(), a_log->dtype(), dt_bias->dtype());
    CHECK_ARGUMENT(out_g->ndim() == 2 && out_beta->ndim() == 2 && a->ndim() == 2 && b->ndim() == 2,
                   "linear_attn_gates: out_g/out_beta/a/b must be 2D");
    CHECK_ARGUMENT(a_log->ndim() == 1 && dt_bias->ndim() == 1,
                   "linear_attn_gates: a_log/dt_bias must be 1D");
    CHECK_ARGUMENT(out_g->shape() == out_beta->shape(), "linear_attn_gates: output shape mismatch");
    CHECK_ARGUMENT(out_g->shape() == a->shape() && out_g->shape() == b->shape(), "linear_attn_gates: input shape mismatch");
    CHECK_ARGUMENT(a_log->shape()[0] == out_g->shape()[1] && dt_bias->shape()[0] == out_g->shape()[1],
                   "linear_attn_gates: vector shape mismatch");
    ASSERT(out_g->isContiguous() && out_beta->isContiguous() && a->isContiguous() && b->isContiguous() && a_log->isContiguous() && dt_bias->isContiguous(),
           "linear_attn_gates: tensors must be contiguous");

    wginfer::core::context().setDevice(out_g->deviceType(), out_g->deviceId());

    switch (out_g->deviceType()) {
    case WGINFER_DEVICE_CPU:
        return cpu::linear_attn_gates(out_g->data(), out_beta->data(), a->data(), b->data(), a_log->data(), dt_bias->data(), out_g->dtype(), out_g->shape()[0], out_g->shape()[1]);
#ifdef ENABLE_NVIDIA_API
    case WGINFER_DEVICE_NVIDIA:
        return nvidia::linear_attn_gates(out_g->data(), out_beta->data(), a->data(), b->data(), a_log->data(), dt_bias->data(), out_g->dtype(), out_g->shape()[0], out_g->shape()[1]);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace wginfer::ops
