#include "op.hpp"

#include "../../core/wginfer_core.hpp"
#include "../../utils.hpp"

#include "./cpu/linear_attn_qkv_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "./nvidia/linear_attn_qkv_nvidia.cuh"
#endif

namespace wginfer::ops {

void linear_attn_qkv_prepare(
    tensor_t out_q,
    tensor_t out_k,
    tensor_t out_v,
    tensor_t mixed_qkv,
    size_t key_heads,
    float eps,
    float q_scale) {
    CHECK_SAME_DEVICE(out_q, out_k, out_v, mixed_qkv);
    CHECK_SAME_DTYPE(out_q->dtype(), out_k->dtype(), out_v->dtype(), mixed_qkv->dtype());
    CHECK_ARGUMENT(out_q->ndim() == 3 && out_k->ndim() == 3 && out_v->ndim() == 3,
                   "linear_attn_qkv_prepare: outputs must be 3D");
    CHECK_ARGUMENT(mixed_qkv->ndim() == 2, "linear_attn_qkv_prepare: mixed_qkv must be 2D");
    CHECK_ARGUMENT(out_q->shape() == out_k->shape(), "linear_attn_qkv_prepare: q/k shape mismatch");
    CHECK_ARGUMENT(out_q->shape()[0] == out_v->shape()[0] && out_q->shape()[1] == out_v->shape()[1],
                   "linear_attn_qkv_prepare: q/v shape mismatch");
    CHECK_ARGUMENT(mixed_qkv->shape()[0] == out_q->shape()[0], "linear_attn_qkv_prepare: seqlen mismatch");
    CHECK_ARGUMENT(key_heads > 0, "linear_attn_qkv_prepare: key_heads must be positive");
    CHECK_ARGUMENT(eps > 0.0f, "linear_attn_qkv_prepare: eps must be positive");

    const size_t seqlen = out_q->shape()[0];
    const size_t value_heads = out_q->shape()[1];
    const size_t key_dim = out_q->shape()[2];
    const size_t value_dim = out_v->shape()[2];
    CHECK_ARGUMENT(key_dim > 0 && value_dim > 0, "linear_attn_qkv_prepare: head dims must be positive");
    CHECK_ARGUMENT(value_heads % key_heads == 0, "linear_attn_qkv_prepare: value_heads must be divisible by key_heads");
    CHECK_ARGUMENT(mixed_qkv->shape()[1] == 2 * key_heads * key_dim + value_heads * value_dim,
                   "linear_attn_qkv_prepare: mixed_qkv width mismatch");
    ASSERT(out_q->isContiguous() && out_k->isContiguous() && out_v->isContiguous() && mixed_qkv->isContiguous(),
           "linear_attn_qkv_prepare: tensors must be contiguous");

    wginfer::core::context().setDevice(out_q->deviceType(), out_q->deviceId());

    switch (out_q->deviceType()) {
    case WGINFER_DEVICE_CPU:
        return cpu::linear_attn_qkv_prepare(
            out_q->data(),
            out_k->data(),
            out_v->data(),
            mixed_qkv->data(),
            out_q->dtype(),
            seqlen,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            eps,
            q_scale);
#ifdef ENABLE_NVIDIA_API
    case WGINFER_DEVICE_NVIDIA:
        return nvidia::linear_attn_qkv_prepare(
            out_q->data(),
            out_k->data(),
            out_v->data(),
            mixed_qkv->data(),
            out_q->dtype(),
            seqlen,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            eps,
            q_scale);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace wginfer::ops
