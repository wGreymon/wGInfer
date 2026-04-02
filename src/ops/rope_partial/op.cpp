#include "op.hpp"

#include "../../core/wginfer_core.hpp"
#include "../../utils.hpp"

#include "./cpu/rope_partial_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "./nvidia/rope_partial_nvidia.cuh"
#endif

namespace wginfer::ops {

void rope_partial(tensor_t out, tensor_t in, tensor_t pos_ids, float theta, size_t rotary_dim) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_ARGUMENT(out->ndim() == 3, "rope_partial: out must be 3D [seqlen, nhead, d]");
    CHECK_ARGUMENT(pos_ids->ndim() == 1, "rope_partial: pos_ids must be 1D [seqlen]");
    CHECK_ARGUMENT(pos_ids->dtype() == WGINFER_DTYPE_I64, "rope_partial: pos_ids must be int64");
    CHECK_ARGUMENT(theta > 0.0f, "rope_partial: theta must be positive");
    CHECK_ARGUMENT(rotary_dim <= out->shape()[2], "rope_partial: rotary_dim out of range");
    CHECK_ARGUMENT((rotary_dim % 2) == 0, "rope_partial: rotary_dim must be even");
    CHECK_ARGUMENT(pos_ids->shape()[0] == out->shape()[0], "rope_partial: pos_ids shape mismatch");
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
           "rope_partial: tensors must be contiguous");

    wginfer::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case WGINFER_DEVICE_CPU:
        return cpu::rope_partial(
            out->data(),
            in->data(),
            reinterpret_cast<const int64_t *>(pos_ids->data()),
            out->dtype(),
            out->shape()[0],
            out->shape()[1],
            out->shape()[2],
            rotary_dim,
            theta);
#ifdef ENABLE_NVIDIA_API
    case WGINFER_DEVICE_NVIDIA:
        return nvidia::rope_partial(
            out->data(),
            in->data(),
            reinterpret_cast<const int64_t *>(pos_ids->data()),
            out->dtype(),
            out->shape()[0],
            out->shape()[1],
            out->shape()[2],
            rotary_dim,
            theta);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace wginfer::ops
