#include "linear_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <vector>

namespace {

float read_value(const std::byte *data, wginferDataType_t dtype, size_t idx) {
    switch (dtype) {
    case WGINFER_DTYPE_F32:
        return reinterpret_cast<const float *>(data)[idx];
    case WGINFER_DTYPE_F16:
        return wginfer::utils::cast<float>(reinterpret_cast<const wginfer::fp16_t *>(data)[idx]);
    case WGINFER_DTYPE_BF16:
        return wginfer::utils::cast<float>(reinterpret_cast<const wginfer::bf16_t *>(data)[idx]);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void write_value(std::byte *data, wginferDataType_t dtype, size_t idx, float value) {
    switch (dtype) {
    case WGINFER_DTYPE_F32:
        reinterpret_cast<float *>(data)[idx] = value;
        return;
    case WGINFER_DTYPE_F16:
        reinterpret_cast<wginfer::fp16_t *>(data)[idx] = wginfer::utils::cast<wginfer::fp16_t>(value);
        return;
    case WGINFER_DTYPE_BF16:
        reinterpret_cast<wginfer::bf16_t *>(data)[idx] = wginfer::utils::cast<wginfer::bf16_t>(value);
        return;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void linear_attention_impl(
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
    std::vector<float> state(nhead * kdim * vdim, 0.0f);
    if (initial_state != nullptr) {
        for (size_t idx = 0; idx < state.size(); ++idx) {
            state[idx] = read_value(initial_state, state_dtype, idx);
        }
    }

    auto state_index = [=](size_t head, size_t ki, size_t vi) {
        return (head * kdim + ki) * vdim + vi;
    };
    auto qkv_index = [=](size_t seq, size_t head, size_t dim, size_t head_dim) {
        return (seq * nhead + head) * head_dim + dim;
    };
    auto scalar_index = [=](size_t seq, size_t head) {
        return seq * nhead + head;
    };

    for (size_t seq = 0; seq < seqlen; ++seq) {
        for (size_t head = 0; head < nhead; ++head) {
            const float g_scale = std::exp(read_value(g, g_dtype, scalar_index(seq, head)));
            const float beta_scale = read_value(beta, beta_dtype, scalar_index(seq, head));

            for (size_t ki = 0; ki < kdim; ++ki) {
                for (size_t vi = 0; vi < vdim; ++vi) {
                    state[state_index(head, ki, vi)] *= g_scale;
                }
            }

            std::vector<float> delta(vdim, 0.0f);
            for (size_t vi = 0; vi < vdim; ++vi) {
                float kv_mem = 0.0f;
                for (size_t ki = 0; ki < kdim; ++ki) {
                    kv_mem +=
                        state[state_index(head, ki, vi)] *
                        read_value(k, data_dtype, qkv_index(seq, head, ki, kdim));
                }
                const float v_val = read_value(v, data_dtype, qkv_index(seq, head, vi, vdim));
                delta[vi] = (v_val - kv_mem) * beta_scale;
            }

            for (size_t ki = 0; ki < kdim; ++ki) {
                const float k_val = read_value(k, data_dtype, qkv_index(seq, head, ki, kdim));
                for (size_t vi = 0; vi < vdim; ++vi) {
                    state[state_index(head, ki, vi)] += k_val * delta[vi];
                }
            }

            for (size_t vi = 0; vi < vdim; ++vi) {
                float acc = 0.0f;
                for (size_t ki = 0; ki < kdim; ++ki) {
                    acc +=
                        state[state_index(head, ki, vi)] *
                        read_value(q, data_dtype, qkv_index(seq, head, ki, kdim));
                }
                write_value(out, data_dtype, qkv_index(seq, head, vi, vdim), acc);
            }
        }
    }

    if (final_state != nullptr) {
        for (size_t idx = 0; idx < state.size(); ++idx) {
            write_value(final_state, state_dtype, idx, state[idx]);
        }
    }
}

} // namespace

namespace wginfer::ops::cpu {

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
    return linear_attention_impl(out, q, k, v, g, beta, initial_state, final_state, data_dtype, g_dtype, beta_dtype, state_dtype, seqlen, nhead, kdim, vdim);
}

} // namespace wginfer::ops::cpu
