#include "op.hpp"

#include "../../core/wginfer_core.hpp"
#include "../../utils.hpp"

#include "./cpu/mul_sigmoid_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "./nvidia/mul_sigmoid_nvidia.cuh"
#endif

namespace wginfer::ops {

void mul_sigmoid(tensor_t out, tensor_t in, tensor_t gate) {
    CHECK_SAME_DEVICE(out, in, gate);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), gate->dtype());
    CHECK_ARGUMENT(out->ndim() == 2 && in->ndim() == 2 && gate->ndim() == 2, "mul_sigmoid: out/in/gate must be 2D");
    CHECK_ARGUMENT(out->shape() == in->shape() && out->shape() == gate->shape(), "mul_sigmoid: shape mismatch");
    ASSERT(out->isContiguous() && in->isContiguous() && gate->isContiguous(), "mul_sigmoid: tensors must be contiguous");

    wginfer::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case WGINFER_DEVICE_CPU:
        return cpu::mul_sigmoid(out->data(), in->data(), gate->data(), out->dtype(), out->shape()[0], out->shape()[1]);
#ifdef ENABLE_NVIDIA_API
    case WGINFER_DEVICE_NVIDIA:
        return nvidia::mul_sigmoid(out->data(), in->data(), gate->data(), out->dtype(), out->shape()[0], out->shape()[1]);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace wginfer::ops
