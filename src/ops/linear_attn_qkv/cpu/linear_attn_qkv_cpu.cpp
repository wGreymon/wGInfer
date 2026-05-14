#include "linear_attn_qkv_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>

namespace {

template <typename T>
void linear_attn_qkv_prepare_impl(
    T *out_q,
    T *out_k,
    T *out_v,
    const T *mixed_qkv,
    size_t seqlen,
    size_t key_heads,
    size_t value_heads,
    size_t key_dim,
    size_t value_dim,
    float eps,
    float q_scale) {
    const size_t repeat_factor = value_heads / key_heads;
    const size_t key_total_dim = key_heads * key_dim;
    const size_t qkv_dim = 2 * key_total_dim + value_heads * value_dim;

    for (size_t seq = 0; seq < seqlen; ++seq) {
        const T *row = mixed_qkv + seq * qkv_dim;
        for (size_t key_head = 0; key_head < key_heads; ++key_head) {
            const T *q_src = row + key_head * key_dim;
            const T *k_src = row + key_total_dim + key_head * key_dim;
            float q_sum_sq = 0.0f;
            float k_sum_sq = 0.0f;
            for (size_t d = 0; d < key_dim; ++d) {
                const float qv = wginfer::utils::cast<float>(q_src[d]);
                const float kv = wginfer::utils::cast<float>(k_src[d]);
                q_sum_sq += qv * qv;
                k_sum_sq += kv * kv;
            }
            const float q_inv_norm = q_scale / std::sqrt(q_sum_sq + eps);
            const float k_inv_norm = 1.0f / std::sqrt(k_sum_sq + eps);
            for (size_t rep = 0; rep < repeat_factor; ++rep) {
                const size_t value_head = key_head * repeat_factor + rep;
                T *q_dst = out_q + (seq * value_heads + value_head) * key_dim;
                T *k_dst = out_k + (seq * value_heads + value_head) * key_dim;
                for (size_t d = 0; d < key_dim; ++d) {
                    q_dst[d] = wginfer::utils::cast<T>(wginfer::utils::cast<float>(q_src[d]) * q_inv_norm);
                    k_dst[d] = wginfer::utils::cast<T>(wginfer::utils::cast<float>(k_src[d]) * k_inv_norm);
                }
            }
        }

        const T *v_src_base = row + 2 * key_total_dim;
        for (size_t value_head = 0; value_head < value_heads; ++value_head) {
            const T *v_src = v_src_base + value_head * value_dim;
            T *v_dst = out_v + (seq * value_heads + value_head) * value_dim;
            std::copy(v_src, v_src + value_dim, v_dst);
        }
    }
}

} // namespace

namespace wginfer::ops::cpu {

void linear_attn_qkv_prepare(
    std::byte *out_q,
    std::byte *out_k,
    std::byte *out_v,
    const std::byte *mixed_qkv,
    wginferDataType_t dtype,
    size_t seqlen,
    size_t key_heads,
    size_t value_heads,
    size_t key_dim,
    size_t value_dim,
    float eps,
    float q_scale) {
    switch (dtype) {
    case WGINFER_DTYPE_F32:
        return linear_attn_qkv_prepare_impl(
            reinterpret_cast<float *>(out_q),
            reinterpret_cast<float *>(out_k),
            reinterpret_cast<float *>(out_v),
            reinterpret_cast<const float *>(mixed_qkv),
            seqlen,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            eps,
            q_scale);
    case WGINFER_DTYPE_F16:
        return linear_attn_qkv_prepare_impl(
            reinterpret_cast<wginfer::fp16_t *>(out_q),
            reinterpret_cast<wginfer::fp16_t *>(out_k),
            reinterpret_cast<wginfer::fp16_t *>(out_v),
            reinterpret_cast<const wginfer::fp16_t *>(mixed_qkv),
            seqlen,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            eps,
            q_scale);
    case WGINFER_DTYPE_BF16:
        return linear_attn_qkv_prepare_impl(
            reinterpret_cast<wginfer::bf16_t *>(out_q),
            reinterpret_cast<wginfer::bf16_t *>(out_k),
            reinterpret_cast<wginfer::bf16_t *>(out_v),
            reinterpret_cast<const wginfer::bf16_t *>(mixed_qkv),
            seqlen,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            eps,
            q_scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace wginfer::ops::cpu
