#include "op.hpp"

#include "../../core/wginfer_core.hpp"
#include "../../utils.hpp"

#include "./cpu/linear_attention_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "./nvidia/linear_attention_nvidia.cuh"
#endif

namespace wginfer::ops {

void linear_attention(
    tensor_t out,
    tensor_t q,
    tensor_t k,
    tensor_t v,
    tensor_t g,
    tensor_t beta,
    tensor_t initial_state,
    tensor_t final_state) {
    CHECK_SAME_DEVICE(out, q, k, v, g, beta);
    CHECK_SAME_DTYPE(out->dtype(), q->dtype(), k->dtype(), v->dtype(), g->dtype(), beta->dtype());
    CHECK_ARGUMENT(out->ndim() == 3, "linear_attention: out must be 3D [seqlen, nhead, vdim]");
    CHECK_ARGUMENT(q->ndim() == 3 && k->ndim() == 3 && v->ndim() == 3, "linear_attention: q, k, v must be 3D");
    CHECK_ARGUMENT(g->ndim() == 2 && beta->ndim() == 2, "linear_attention: g and beta must be 2D [seqlen, nhead]");
    CHECK_ARGUMENT(q->shape()[0] == k->shape()[0] && q->shape()[0] == v->shape()[0], "linear_attention: sequence lengths must match");
    CHECK_ARGUMENT(q->shape()[1] == k->shape()[1] && q->shape()[1] == v->shape()[1], "linear_attention: number of heads must match");
    CHECK_ARGUMENT(out->shape()[0] == q->shape()[0] && out->shape()[1] == q->shape()[1], "linear_attention: out shape must match sequence/head dimensions");
    CHECK_ARGUMENT(out->shape()[2] == v->shape()[2], "linear_attention: out vdim must match v");
    CHECK_ARGUMENT(q->shape()[2] == k->shape()[2], "linear_attention: q and k head dimensions must match");
    CHECK_ARGUMENT(g->shape()[0] == q->shape()[0] && g->shape()[1] == q->shape()[1], "linear_attention: g shape must match [seqlen, nhead]");
    CHECK_ARGUMENT(beta->shape()[0] == q->shape()[0] && beta->shape()[1] == q->shape()[1], "linear_attention: beta shape must match [seqlen, nhead]");
    ASSERT(out->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous() &&
               g->isContiguous() && beta->isContiguous(),
           "linear_attention: all input/output tensors must be contiguous");

    if (initial_state != nullptr) {
        CHECK_SAME_DEVICE(out, initial_state);
        CHECK_ARGUMENT(initial_state->ndim() == 3, "linear_attention: initial_state must be 3D [nhead, kdim, vdim]");
        CHECK_ARGUMENT(initial_state->shape()[0] == q->shape()[1], "linear_attention: initial_state nhead mismatch");
        CHECK_ARGUMENT(initial_state->shape()[1] == q->shape()[2], "linear_attention: initial_state kdim mismatch");
        CHECK_ARGUMENT(initial_state->shape()[2] == v->shape()[2], "linear_attention: initial_state vdim mismatch");
        CHECK_ARGUMENT(initial_state->dtype() == out->dtype(), "linear_attention: initial_state dtype mismatch");
        ASSERT(initial_state->isContiguous(), "linear_attention: initial_state must be contiguous");
    }

    if (final_state != nullptr) {
        CHECK_SAME_DEVICE(out, final_state);
        CHECK_ARGUMENT(final_state->ndim() == 3, "linear_attention: final_state must be 3D [nhead, kdim, vdim]");
        CHECK_ARGUMENT(final_state->shape()[0] == q->shape()[1], "linear_attention: final_state nhead mismatch");
        CHECK_ARGUMENT(final_state->shape()[1] == q->shape()[2], "linear_attention: final_state kdim mismatch");
        CHECK_ARGUMENT(final_state->shape()[2] == v->shape()[2], "linear_attention: final_state vdim mismatch");
        CHECK_ARGUMENT(final_state->dtype() == out->dtype(), "linear_attention: final_state dtype mismatch");
        ASSERT(final_state->isContiguous(), "linear_attention: final_state must be contiguous");
    }

    wginfer::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case WGINFER_DEVICE_CPU:
        return cpu::linear_attention(
            out->data(),
            q->data(),
            k->data(),
            v->data(),
            g->data(),
            beta->data(),
            initial_state ? initial_state->data() : nullptr,
            final_state ? final_state->data() : nullptr,
            out->dtype(),
            out->shape()[0],
            out->shape()[1],
            q->shape()[2],
            v->shape()[2]);
#ifdef ENABLE_NVIDIA_API
    case WGINFER_DEVICE_NVIDIA:
        return nvidia::linear_attention(
            out->data(),
            q->data(),
            k->data(),
            v->data(),
            g->data(),
            beta->data(),
            initial_state ? initial_state->data() : nullptr,
            final_state ? final_state->data() : nullptr,
            out->dtype(),
            out->shape()[0],
            out->shape()[1],
            q->shape()[2],
            v->shape()[2]);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace wginfer::ops
