#include "linear_attention_nvidia.cuh"

#include "../../../utils.hpp"
#include "../../../utils/gpu_utils.hpp"

#include <cmath>

namespace {

template <typename T>
__global__ void linear_attention_kernel(
    T *out,
    const T *q,
    const T *k,
    const T *v,
    const T *g,
    const T *beta,
    const T *initial_state,
    T *state,
    size_t seqlen,
    size_t nhead,
    size_t kdim,
    size_t vdim) {
    const size_t head = static_cast<size_t>(blockIdx.x);
    if (head >= nhead || threadIdx.x != 0) {
        return;
    }

    const size_t state_elems_per_head = kdim * vdim;
    T *state_head = state + head * state_elems_per_head;
    const T *initial_state_head = initial_state ? (initial_state + head * state_elems_per_head) : nullptr;

    for (size_t idx = 0; idx < state_elems_per_head; ++idx) {
        state_head[idx] = initial_state_head ? initial_state_head[idx] : from_float<T>(0.0f);
    }

    for (size_t seq = 0; seq < seqlen; ++seq) {
        const size_t scalar_offset = seq * nhead + head;
        const float g_scale = expf(to_float(g[scalar_offset]));
        const float beta_scale = to_float(beta[scalar_offset]);

        for (size_t ki = 0; ki < kdim; ++ki) {
            for (size_t vi = 0; vi < vdim; ++vi) {
                const size_t idx = ki * vdim + vi;
                state_head[idx] = from_float<T>(to_float(state_head[idx]) * g_scale);
            }
        }

        for (size_t vi = 0; vi < vdim; ++vi) {
            float kv_mem = 0.0f;
            for (size_t ki = 0; ki < kdim; ++ki) {
                const size_t state_idx = ki * vdim + vi;
                const size_t qk_offset = (seq * nhead + head) * kdim + ki;
                kv_mem += to_float(state_head[state_idx]) * to_float(k[qk_offset]);
            }
            const size_t v_offset = (seq * nhead + head) * vdim + vi;
            const float v_val = to_float(v[v_offset]);
            const float delta = (v_val - kv_mem) * beta_scale;
            for (size_t ki = 0; ki < kdim; ++ki) {
                const size_t idx = ki * vdim + vi;
                const size_t qk_offset = (seq * nhead + head) * kdim + ki;
                const float updated = to_float(state_head[idx]) + to_float(k[qk_offset]) * delta;
                state_head[idx] = from_float<T>(updated);
            }
        }

        for (size_t vi = 0; vi < vdim; ++vi) {
            float acc = 0.0f;
            for (size_t ki = 0; ki < kdim; ++ki) {
                const size_t state_idx = ki * vdim + vi;
                const size_t q_offset = (seq * nhead + head) * kdim + ki;
                acc += to_float(state_head[state_idx]) * to_float(q[q_offset]);
            }
            const size_t out_offset = (seq * nhead + head) * vdim + vi;
            out[out_offset] = from_float<T>(acc);
        }
    }
}

template <typename T>
void launch_linear_attention(
    std::byte *out,
    const std::byte *q,
    const std::byte *k,
    const std::byte *v,
    const std::byte *g,
    const std::byte *beta,
    const std::byte *initial_state,
    std::byte *final_state,
    size_t seqlen,
    size_t nhead,
    size_t kdim,
    size_t vdim) {
    const size_t state_bytes = nhead * kdim * vdim * sizeof(T);
    T *state_buffer = reinterpret_cast<T *>(final_state);
    bool owns_state_buffer = false;
    if (state_buffer == nullptr) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state_buffer), state_bytes));
        owns_state_buffer = true;
    }

    linear_attention_kernel<T><<<static_cast<unsigned int>(nhead), 1>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(q),
        reinterpret_cast<const T *>(k),
        reinterpret_cast<const T *>(v),
        reinterpret_cast<const T *>(g),
        reinterpret_cast<const T *>(beta),
        reinterpret_cast<const T *>(initial_state),
        state_buffer,
        seqlen,
        nhead,
        kdim,
        vdim);
    CUDA_CHECK(cudaGetLastError());

    if (owns_state_buffer) {
        CUDA_CHECK(cudaFree(state_buffer));
    }
}

} // namespace

namespace wginfer::ops::nvidia {

void linear_attention(
    std::byte *out,
    const std::byte *q,
    const std::byte *k,
    const std::byte *v,
    const std::byte *g,
    const std::byte *beta,
    const std::byte *initial_state,
    std::byte *final_state,
    wginferDataType_t dtype,
    size_t seqlen,
    size_t nhead,
    size_t kdim,
    size_t vdim) {
    if (seqlen == 0 || nhead == 0 || kdim == 0 || vdim == 0) {
        return;
    }

    switch (dtype) {
    case WGINFER_DTYPE_F32:
        launch_linear_attention<float>(out, q, k, v, g, beta, initial_state, final_state, seqlen, nhead, kdim, vdim);
        break;
    case WGINFER_DTYPE_F16:
        launch_linear_attention<half>(out, q, k, v, g, beta, initial_state, final_state, seqlen, nhead, kdim, vdim);
        break;
    case WGINFER_DTYPE_BF16:
        launch_linear_attention<__nv_bfloat16>(out, q, k, v, g, beta, initial_state, final_state, seqlen, nhead, kdim, vdim);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace wginfer::ops::nvidia
