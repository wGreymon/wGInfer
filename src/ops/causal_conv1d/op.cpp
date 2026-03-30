#include "op.hpp"

#include "../../core/wginfer_core.hpp"
#include "../../utils.hpp"

#include "./cpu/causal_conv1d_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "./nvidia/causal_conv1d_nvidia.cuh"
#endif

namespace wginfer::ops {

void causal_conv1d(tensor_t out, tensor_t in, tensor_t weight) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    CHECK_ARGUMENT(out->ndim() == 2 && in->ndim() == 2, "causal_conv1d: out and in must be 2D [seqlen, channels]");
    CHECK_ARGUMENT(weight->ndim() == 3, "causal_conv1d: weight must be 3D [channels, 1, kernel_size]");
    CHECK_ARGUMENT(out->shape() == in->shape(), "causal_conv1d: out and in shape mismatch");
    CHECK_ARGUMENT(weight->shape()[0] == in->shape()[1], "causal_conv1d: weight channels mismatch");
    CHECK_ARGUMENT(weight->shape()[1] == 1, "causal_conv1d: weight middle dimension must be 1");
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "causal_conv1d: tensors must be contiguous");

    wginfer::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case WGINFER_DEVICE_CPU:
        return cpu::causal_conv1d(
            out->data(),
            in->data(),
            weight->data(),
            out->dtype(),
            out->shape()[0],
            out->shape()[1],
            weight->shape()[2]);
#ifdef ENABLE_NVIDIA_API
    case WGINFER_DEVICE_NVIDIA:
        return nvidia::causal_conv1d(
            out->data(),
            in->data(),
            weight->data(),
            out->dtype(),
            out->shape()[0],
            out->shape()[1],
            weight->shape()[2]);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace wginfer::ops
