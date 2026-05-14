#include "linear_attention_nvidia.cuh"

#include "../../../utils.hpp"
#include "../../../utils/gpu_utils.hpp"

#include <cmath>

namespace {

__device__ __forceinline__ float read_value(const std::byte *data, wginferDataType_t dtype, size_t idx) {
    switch (dtype) {
    case WGINFER_DTYPE_F32:
        return reinterpret_cast<const float *>(data)[idx];
    case WGINFER_DTYPE_F16:
        return __half2float(reinterpret_cast<const half *>(data)[idx]);
    case WGINFER_DTYPE_BF16:
        return __bfloat162float(reinterpret_cast<const __nv_bfloat16 *>(data)[idx]);
    default:
        return 0.0f;
    }
}

template <typename DataT, typename StateT>
__global__ void linear_attention_kernel(
    DataT *out,
    const DataT *q,
    const DataT *k,
    const DataT *v,
    const std::byte *g,
    const std::byte *beta,
    wginferDataType_t g_dtype,
    wginferDataType_t beta_dtype,
    const StateT *initial_state,
    StateT *state,
    size_t seqlen,
    size_t nhead,
    size_t kdim,
    size_t vdim) {
    const size_t head = static_cast<size_t>(blockIdx.x);
    if (head >= nhead || threadIdx.x != 0) {
        return;
    }

    const size_t state_elems_per_head = kdim * vdim;
    StateT *state_head = state + head * state_elems_per_head;
    const StateT *initial_state_head = initial_state ? (initial_state + head * state_elems_per_head) : nullptr;

    for (size_t idx = 0; idx < state_elems_per_head; ++idx) {
        state_head[idx] = initial_state_head ? initial_state_head[idx] : from_float<StateT>(0.0f);
    }

    for (size_t seq = 0; seq < seqlen; ++seq) {
        const size_t scalar_offset = seq * nhead + head;
        const float g_scale = expf(read_value(g, g_dtype, scalar_offset));
        const float beta_scale = read_value(beta, beta_dtype, scalar_offset);

        for (size_t ki = 0; ki < kdim; ++ki) {
            for (size_t vi = 0; vi < vdim; ++vi) {
                const size_t idx = ki * vdim + vi;
                state_head[idx] = from_float<StateT>(to_float(state_head[idx]) * g_scale);
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
                state_head[idx] = from_float<StateT>(updated);
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
            out[out_offset] = from_float<DataT>(acc);
        }
    }
}

template <typename DataT, typename StateT>
void launch_linear_attention(
    std::byte *out,
    const std::byte *q,
    const std::byte *k,
    const std::byte *v,
    const std::byte *g,
    const std::byte *beta,
    const std::byte *initial_state,
    std::byte *final_state,
    wginferDataType_t g_dtype,
    wginferDataType_t beta_dtype,
    size_t seqlen,
    size_t nhead,
    size_t kdim,
    size_t vdim) {
    const size_t state_bytes = nhead * kdim * vdim * sizeof(StateT);
    StateT *state_buffer = reinterpret_cast<StateT *>(final_state);
    bool owns_state_buffer = false;
    if (state_buffer == nullptr) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state_buffer), state_bytes));
        owns_state_buffer = true;
    }

    linear_attention_kernel<DataT, StateT><<<static_cast<unsigned int>(nhead), 1>>>(
        reinterpret_cast<DataT *>(out),
        reinterpret_cast<const DataT *>(q),
        reinterpret_cast<const DataT *>(k),
        reinterpret_cast<const DataT *>(v),
        g,
        beta,
        g_dtype,
        beta_dtype,
        reinterpret_cast<const StateT *>(initial_state),
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

template <typename DataT>
void dispatch_state_dtype(
    std::byte *out,
    const std::byte *q,
    const std::byte *k,
    const std::byte *v,
    const std::byte *g,
    const std::byte *beta,
    const std::byte *initial_state,
    std::byte *final_state,
    wginferDataType_t g_dtype,
    wginferDataType_t beta_dtype,
    wginferDataType_t state_dtype,
    size_t seqlen,
    size_t nhead,
    size_t kdim,
    size_t vdim) {
    switch (state_dtype) {
    case WGINFER_DTYPE_F32:
        return launch_linear_attention<DataT, float>(out, q, k, v, g, beta, initial_state, final_state, g_dtype, beta_dtype, seqlen, nhead, kdim, vdim);
    case WGINFER_DTYPE_F16:
        return launch_linear_attention<DataT, half>(out, q, k, v, g, beta, initial_state, final_state, g_dtype, beta_dtype, seqlen, nhead, kdim, vdim);
    case WGINFER_DTYPE_BF16:
        return launch_linear_attention<DataT, __nv_bfloat16>(out, q, k, v, g, beta, initial_state, final_state, g_dtype, beta_dtype, seqlen, nhead, kdim, vdim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(state_dtype);
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
    wginferDataType_t data_dtype,
    wginferDataType_t g_dtype,
    wginferDataType_t beta_dtype,
    wginferDataType_t state_dtype,
    size_t seqlen,
    size_t nhead,
    size_t kdim,
    size_t vdim) {
    if (seqlen == 0 || nhead == 0 || kdim == 0 || vdim == 0) {
        return;
    }

    switch (data_dtype) {
    case WGINFER_DTYPE_F32:
        return dispatch_state_dtype<float>(out, q, k, v, g, beta, initial_state, final_state, g_dtype, beta_dtype, state_dtype, seqlen, nhead, kdim, vdim);
    case WGINFER_DTYPE_F16:
        return dispatch_state_dtype<half>(out, q, k, v, g, beta, initial_state, final_state, g_dtype, beta_dtype, state_dtype, seqlen, nhead, kdim, vdim);
    case WGINFER_DTYPE_BF16:
        return dispatch_state_dtype<__nv_bfloat16>(out, q, k, v, g, beta, initial_state, final_state, g_dtype, beta_dtype, state_dtype, seqlen, nhead, kdim, vdim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(data_dtype);
    }
}

} // namespace wginfer::ops::nvidia
