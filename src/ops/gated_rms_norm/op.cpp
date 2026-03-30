#include "op.hpp"

#include "../../core/wginfer_core.hpp"
#include "../../utils.hpp"

#include "./cpu/gated_rms_norm_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "./nvidia/gated_rms_norm_nvidia.cuh"
#endif

namespace wginfer::ops {

void gated_rms_norm(tensor_t out, tensor_t in, tensor_t gate, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, gate, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), gate->dtype(), weight->dtype());
    CHECK_ARGUMENT(out->ndim() == 2 && in->ndim() == 2 && gate->ndim() == 2, "gated_rms_norm: out/in/gate must be 2D");
    CHECK_ARGUMENT(weight->ndim() == 1, "gated_rms_norm: weight must be 1D");
    CHECK_ARGUMENT(out->shape() == in->shape(), "gated_rms_norm: out and in shape mismatch");
    CHECK_ARGUMENT(out->shape() == gate->shape(), "gated_rms_norm: out and gate shape mismatch");
    CHECK_ARGUMENT(weight->shape()[0] == out->shape()[1], "gated_rms_norm: weight size mismatch");
    ASSERT(out->isContiguous() && in->isContiguous() && gate->isContiguous() && weight->isContiguous(),
           "gated_rms_norm: tensors must be contiguous");

    wginfer::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case WGINFER_DEVICE_CPU:
        return cpu::gated_rms_norm(
            out->data(),
            in->data(),
            gate->data(),
            weight->data(),
            out->dtype(),
            out->shape()[0],
            out->shape()[1],
            eps);
#ifdef ENABLE_NVIDIA_API
    case WGINFER_DEVICE_NVIDIA:
        return nvidia::gated_rms_norm(
            out->data(),
            in->data(),
            gate->data(),
            weight->data(),
            out->dtype(),
            out->shape()[0],
            out->shape()[1],
            eps);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace wginfer::ops
