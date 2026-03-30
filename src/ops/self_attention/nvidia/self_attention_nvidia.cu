#include "self_attention_nvidia.cuh"

#include "../../../utils.hpp"
#include "../../../utils/gpu_utils.hpp"

#include <cstddef>

namespace {

template <typename T, int BLOCK_Q, int BLOCK_KV>
__global__ void flash_attention_kernel(T *__restrict__ out,
                                       const T *__restrict__ q,
                                       const T *__restrict__ k,
                                       const T *__restrict__ v,
                                       size_t seqlen,
                                       size_t nhead,
                                       size_t nkvhead,
                                       size_t d,
                                       size_t dv,
                                       size_t total_len,
                                       float scale) {
    const size_t qh = static_cast<size_t>(blockIdx.y);
    const size_t q_block_idx = static_cast<size_t>(blockIdx.x);
    if (qh >= nhead) {
        return;
    }

    const size_t q_start = q_block_idx * BLOCK_Q;
    if (q_start >= seqlen) {
        return;
    }

    const size_t q_end = min(q_start + static_cast<size_t>(BLOCK_Q), seqlen);
    const size_t q_block_len = q_end - q_start;
    const size_t kv_head = qh * nkvhead / nhead;     // GQA映射
    const ptrdiff_t diag = static_cast<ptrdiff_t>(total_len) - static_cast<ptrdiff_t>(seqlen);

    extern __shared__ float smem[];
    size_t offset = 0;

    float *q_shared = smem + offset;
    offset += BLOCK_Q * d;

    float *k_shared = smem + offset;
    offset += BLOCK_KV * d;

    float *v_shared = smem + offset;
    offset += BLOCK_KV * dv;

    float *s_shared = smem + offset;
    offset += BLOCK_Q * BLOCK_KV;

    float *o_shared = smem + offset;
    offset += BLOCK_Q * dv;

    float *m_shared = smem + offset;
    offset += BLOCK_Q;

    float *l_shared = smem + offset;

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    for (size_t idx = static_cast<size_t>(tid); idx < q_block_len; idx += static_cast<size_t>(num_threads)) {
        m_shared[idx] = -INFINITY;
        l_shared[idx] = 0.0f;
    }
    for (size_t idx = static_cast<size_t>(tid); idx < q_block_len * dv; idx += static_cast<size_t>(num_threads)) {
        o_shared[idx] = 0.0f;
    }
    __syncthreads();

    for (size_t idx = static_cast<size_t>(tid); idx < q_block_len * d; idx += static_cast<size_t>(num_threads)) {
        const size_t q_idx = idx / d;
        const size_t d_idx = idx % d;
        const T *q_row = q + ((q_start + q_idx) * nhead + qh) * d;
        q_shared[q_idx * d + d_idx] = to_float(q_row[d_idx]);
    }
    __syncthreads();

    for (size_t kv_start = 0; kv_start < total_len; kv_start += BLOCK_KV) {
        const size_t kv_end = min(kv_start + static_cast<size_t>(BLOCK_KV), total_len);
        const size_t kv_block_len = kv_end - kv_start;

        for (size_t idx = static_cast<size_t>(tid); idx < kv_block_len * d; idx += static_cast<size_t>(num_threads)) {
            const size_t kv_idx = idx / d;
            const size_t d_idx = idx % d;
            const T *k_row = k + ((kv_start + kv_idx) * nkvhead + kv_head) * d;
            k_shared[kv_idx * d + d_idx] = to_float(k_row[d_idx]);
        }
        for (size_t idx = static_cast<size_t>(tid); idx < kv_block_len * dv; idx += static_cast<size_t>(num_threads)) {
            const size_t kv_idx = idx / dv;
            const size_t d_idx = idx % dv;
            const T *v_row = v + ((kv_start + kv_idx) * nkvhead + kv_head) * dv;
            v_shared[kv_idx * dv + d_idx] = to_float(v_row[d_idx]);
        }
        __syncthreads();

        for (size_t idx = static_cast<size_t>(tid); idx < q_block_len * kv_block_len; idx += static_cast<size_t>(num_threads)) {
            const size_t q_idx = idx / kv_block_len;
            const size_t kv_idx = idx % kv_block_len;
            const ptrdiff_t max_visible_key = static_cast<ptrdiff_t>(q_start + q_idx) + diag;
            const size_t score_offset = q_idx * BLOCK_KV + kv_idx;
            if (static_cast<ptrdiff_t>(kv_start + kv_idx) > max_visible_key) {
                s_shared[score_offset] = -INFINITY;
                continue;
            }

            float dot = 0.0f;
            const size_t q_base = q_idx * d;
            const size_t k_base = kv_idx * d;
            for (size_t kd = 0; kd < d; ++kd) {
                dot += q_shared[q_base + kd] * k_shared[k_base + kd];
            }
            s_shared[score_offset] = dot * scale;
        }
        __syncthreads();

        for (size_t q_idx = static_cast<size_t>(tid); q_idx < q_block_len; q_idx += static_cast<size_t>(num_threads)) {
            float m_old = m_shared[q_idx];
            float l_old = l_shared[q_idx];
            float m_ij = -INFINITY;
            const size_t score_row = q_idx * BLOCK_KV;

            for (size_t kv_idx = 0; kv_idx < kv_block_len; ++kv_idx) {
                m_ij = fmaxf(m_ij, s_shared[score_row + kv_idx]);
            }

            const float m_new = fmaxf(m_old, m_ij);
            if (!isfinite(m_new)) {
                continue;
            }

            const float alpha = (l_old == 0.0f) ? 0.0f : expf(m_old - m_new);
            float p_sum = 0.0f;
            for (size_t kv_idx = 0; kv_idx < kv_block_len; ++kv_idx) {
                const float p_ij = expf(s_shared[score_row + kv_idx] - m_new);
                s_shared[score_row + kv_idx] = p_ij;
                p_sum += p_ij;
            }

            const size_t out_base = q_idx * dv;
            for (size_t vd = 0; vd < dv; ++vd) {
                float acc = alpha * o_shared[out_base + vd];
                for (size_t kv_idx = 0; kv_idx < kv_block_len; ++kv_idx) {
                    acc += s_shared[score_row + kv_idx] * v_shared[kv_idx * dv + vd];
                }
                o_shared[out_base + vd] = acc;
            }

            m_shared[q_idx] = m_new;
            l_shared[q_idx] = alpha * l_old + p_sum;
        }
        __syncthreads();
    }

    for (size_t idx = static_cast<size_t>(tid); idx < q_block_len * dv; idx += static_cast<size_t>(num_threads)) {
        const size_t q_idx = idx / dv;
        const size_t d_idx = idx % dv;
        const float l = l_shared[q_idx];
        const float out_val = (l > 0.0f) ? (o_shared[idx] / l) : 0.0f;
        T *out_row = out + ((q_start + q_idx) * nhead + qh) * dv;
        out_row[d_idx] = from_float<T>(out_val);
    }
}

} // namespace

namespace wginfer::ops::nvidia {

template <typename T>
static void launch_flash_attention(std::byte *attn_val,
                                   const std::byte *q,
                                   const std::byte *k,
                                   const std::byte *v,
                                   size_t seqlen,
                                   size_t nhead,
                                   size_t nkvhead,
                                   size_t d,
                                   size_t dv,
                                   size_t total_len,
                                   float scale) {
    constexpr int block_q = 8;
    constexpr int block_kv = 16;
    constexpr int block_size = 128;
    const dim3 grid_dim(CEIL(seqlen, static_cast<size_t>(block_q)), nhead, 1);
    const size_t smem_bytes = sizeof(float) * (block_q * d + block_kv * d + block_kv * dv +
                                               block_q * block_kv + block_q * dv + 2 * block_q);

    flash_attention_kernel<T, block_q, block_kv><<<grid_dim, block_size, smem_bytes>>>(
        reinterpret_cast<T *>(attn_val),
        reinterpret_cast<const T *>(q),
        reinterpret_cast<const T *>(k),
        reinterpret_cast<const T *>(v),
        seqlen,
        nhead,
        nkvhead,
        d,
        dv,
        total_len,
        scale);
}

void self_attention(std::byte *attn_val,
                    const std::byte *q,
                    const std::byte *k,
                    const std::byte *v,
                    wginferDataType_t type,
                    size_t seqlen,
                    size_t nhead,
                    size_t nkvhead,
                    size_t d,
                    size_t dv,
                    size_t total_len,
                    float scale) {
    if (seqlen == 0 || nhead == 0 || nkvhead == 0 || d == 0 || dv == 0 || total_len == 0) {
        return;
    }

    switch (type) {
    case WGINFER_DTYPE_F32:
        launch_flash_attention<float>(attn_val, q, k, v, seqlen, nhead, nkvhead, d, dv, total_len, scale);
        break;
    case WGINFER_DTYPE_F16:
        launch_flash_attention<half>(attn_val, q, k, v, seqlen, nhead, nkvhead, d, dv, total_len, scale);
        break;
    case WGINFER_DTYPE_BF16:
        launch_flash_attention<__nv_bfloat16>(attn_val, q, k, v, seqlen, nhead, nkvhead, d, dv, total_len, scale);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    CUDA_CHECK(cudaGetLastError());
}

} // namespace wginfer::ops::nvidia
