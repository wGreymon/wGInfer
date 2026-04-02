#include "op.hpp"

#include "../../core/wginfer_core.hpp"
#include "../../utils.hpp"

#include "./cpu/normalize_l2_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "./nvidia/normalize_l2_nvidia.cuh"
#endif

namespace wginfer::ops {

void normalize_l2(tensor_t out, tensor_t in, float eps, float scale) {
    CHECK_SAME_DEVICE(out, in);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_ARGUMENT(out->ndim() == 2 && in->ndim() == 2, "normalize_l2: out/in must be 2D");
    CHECK_ARGUMENT(out->shape() == in->shape(), "normalize_l2: out and in shape mismatch");
    CHECK_ARGUMENT(eps > 0.0f, "normalize_l2: eps must be positive");
    ASSERT(out->isContiguous() && in->isContiguous(), "normalize_l2: tensors must be contiguous");

    wginfer::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case WGINFER_DEVICE_CPU:
        return cpu::normalize_l2(
            out->data(),
            in->data(),
            out->dtype(),
            out->shape()[0],
            out->shape()[1],
            eps,
            scale);
#ifdef ENABLE_NVIDIA_API
    case WGINFER_DEVICE_NVIDIA:
        return nvidia::normalize_l2(
            out->data(),
            in->data(),
            out->dtype(),
            out->shape()[0],
            out->shape()[1],
            eps,
            scale);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace wginfer::ops
