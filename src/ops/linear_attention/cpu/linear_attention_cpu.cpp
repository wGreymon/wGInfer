#include "linear_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <vector>

namespace {

template <typename T>
void linear_attention_impl(
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
    const T *q_t = reinterpret_cast<const T *>(q);
    const T *k_t = reinterpret_cast<const T *>(k);
    const T *v_t = reinterpret_cast<const T *>(v);
    const T *g_t = reinterpret_cast<const T *>(g);
    const T *beta_t = reinterpret_cast<const T *>(beta);
    const T *initial_state_t = reinterpret_cast<const T *>(initial_state);
    T *out_t = reinterpret_cast<T *>(out);
    T *final_state_t = reinterpret_cast<T *>(final_state);

    std::vector<float> state(nhead * kdim * vdim, 0.0f);
    if (initial_state_t != nullptr) {
        for (size_t idx = 0; idx < state.size(); ++idx) {
            state[idx] = wginfer::utils::cast<float>(initial_state_t[idx]);
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
            const float g_scale = std::exp(wginfer::utils::cast<float>(g_t[scalar_index(seq, head)]));
            const float beta_scale = wginfer::utils::cast<float>(beta_t[scalar_index(seq, head)]);

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
                        wginfer::utils::cast<float>(k_t[qkv_index(seq, head, ki, kdim)]);
                }
                const float v_val = wginfer::utils::cast<float>(v_t[qkv_index(seq, head, vi, vdim)]);
                delta[vi] = (v_val - kv_mem) * beta_scale;
            }

            for (size_t ki = 0; ki < kdim; ++ki) {
                const float k_val = wginfer::utils::cast<float>(k_t[qkv_index(seq, head, ki, kdim)]);
                for (size_t vi = 0; vi < vdim; ++vi) {
                    state[state_index(head, ki, vi)] += k_val * delta[vi];
                }
            }

            for (size_t vi = 0; vi < vdim; ++vi) {
                float acc = 0.0f;
                for (size_t ki = 0; ki < kdim; ++ki) {
                    acc +=
                        state[state_index(head, ki, vi)] *
                        wginfer::utils::cast<float>(q_t[qkv_index(seq, head, ki, kdim)]);
                }
                out_t[qkv_index(seq, head, vi, vdim)] = wginfer::utils::cast<T>(acc);
            }
        }
    }

    if (final_state_t != nullptr) {
        for (size_t idx = 0; idx < state.size(); ++idx) {
            final_state_t[idx] = wginfer::utils::cast<T>(state[idx]);
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
    wginferDataType_t dtype,
    size_t seqlen,
    size_t nhead,
    size_t kdim,
    size_t vdim) {
    switch (dtype) {
    case WGINFER_DTYPE_F32:
        return linear_attention_impl<float>(out, q, k, v, g, beta, initial_state, final_state, seqlen, nhead, kdim, vdim);
    case WGINFER_DTYPE_F16:
        return linear_attention_impl<wginfer::fp16_t>(out, q, k, v, g, beta, initial_state, final_state, seqlen, nhead, kdim, vdim);
    case WGINFER_DTYPE_BF16:
        return linear_attention_impl<wginfer::bf16_t>(out, q, k, v, g, beta, initial_state, final_state, seqlen, nhead, kdim, vdim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace wginfer::ops::cpu
