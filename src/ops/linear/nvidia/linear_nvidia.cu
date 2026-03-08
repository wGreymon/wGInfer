#include "linear_nvidia.cuh"

#include "../../../utils.hpp"
#include "../../../utils/gpu_utils.hpp"
#include <cublas_v2.h>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace {

template <typename T>
__device__ __forceinline__ bool is_aligned_16(const T *ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) & 0xF) == 0;
}

inline void cublas_check(cublasStatus_t status, const char *msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(msg);
    }
}

inline cublasHandle_t get_cublas_handle() {
    static thread_local cublasHandle_t handle = []() {
        cublasHandle_t h = nullptr;
        cublas_check(cublasCreate(&h), "cublasCreate failed");
        return h;
    }();
    return handle;
}

template <typename T>
__global__ void add_bias_rowwise_kernel(T *out, const T *bias, size_t M, size_t N) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = M * N;
    for (size_t i = idx; i < total; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const size_t col = i % N;
        out[i] = from_float<T>(to_float(out[i]) + to_float(bias[col]));
    }
}

template <typename T>
inline void launch_add_bias(T *out, const T *bias, size_t M, size_t N) {
    if (bias == nullptr || M == 0 || N == 0) {
        return;
    }
    constexpr int block_size = 256;
    const int grid_size = static_cast<int>(CEIL(M * N, block_size));
    add_bias_rowwise_kernel<<<grid_size, block_size>>>(out, bias, M, N);
}

inline void linear_cublas_f32(float *out, const float *in, const float *weight,
                              const float *bias, size_t M, size_t N, size_t K) {
    cublasHandle_t handle = get_cublas_handle();
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // Row-major: out[M,N] = in[M,K] * weight[N,K]^T
    // Column-major mapping: C[N,M] = A[N,K] * B[K,M], where A=weight^T(op=T), B=in(op=N).
    cublas_check(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(N),
                             static_cast<int>(M), static_cast<int>(K), &alpha, weight,
                             static_cast<int>(K), in, static_cast<int>(K), &beta, out,
                             static_cast<int>(N)),
                 "cublasSgemm failed");
    launch_add_bias(out, bias, M, N);
}

inline void linear_cublas_f16(half *out, const half *in, const half *weight,
                              const half *bias, size_t M, size_t N, size_t K) {
    cublasHandle_t handle = get_cublas_handle();
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t status = cublasGemmEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(N), static_cast<int>(M),
        static_cast<int>(K), &alpha, weight, CUDA_R_16F, static_cast<int>(K), in,
        CUDA_R_16F, static_cast<int>(K), &beta, out, CUDA_R_16F, static_cast<int>(N),
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (status == CUBLAS_STATUS_NOT_SUPPORTED) {
        status = cublasGemmEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(N),
            static_cast<int>(M), static_cast<int>(K), &alpha, weight, CUDA_R_16F,
            static_cast<int>(K), in, CUDA_R_16F, static_cast<int>(K), &beta, out,
            CUDA_R_16F, static_cast<int>(N), CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT);
    }
    cublas_check(status, "cublasGemmEx f16 failed");
    launch_add_bias(out, bias, M, N);
}

inline void linear_cublas_bf16(__nv_bfloat16 *out, const __nv_bfloat16 *in,
                               const __nv_bfloat16 *weight,
                               const __nv_bfloat16 *bias, size_t M, size_t N,
                               size_t K) {
    cublasHandle_t handle = get_cublas_handle();
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t status = cublasGemmEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(N), static_cast<int>(M),
        static_cast<int>(K), &alpha, weight, CUDA_R_16BF, static_cast<int>(K), in,
        CUDA_R_16BF, static_cast<int>(K), &beta, out, CUDA_R_16BF,
        static_cast<int>(N), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (status == CUBLAS_STATUS_NOT_SUPPORTED) {
        status = cublasGemmEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(N),
            static_cast<int>(M), static_cast<int>(K), &alpha, weight, CUDA_R_16BF,
            static_cast<int>(K), in, CUDA_R_16BF, static_cast<int>(K), &beta, out,
            CUDA_R_16BF, static_cast<int>(N), CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT);
    }
    cublas_check(status, "cublasGemmEx bf16 failed");
    launch_add_bias(out, bias, M, N);
}

// cpu_time:
// Torch time: 30.81158 ms
// WGINFER time: 401.65733 ms
// Torch time: 140.67506 ms
// WGINFER time: 3028.21840 ms
// Torch time: 142.86126 ms
// WGINFER time: 2105.92961 ms

// naive：使用global memory实现
// in[M, K], weight[N, K], bias[N], out[M, N]
// v1_time:
// Torch time: 2.06076 ms
// WGINFER time: 82.52521 ms
// Torch time: 0.58656 ms
// WGINFER time: 82.01252 ms
// Torch time: 0.59076 ms
// WGINFER time: 82.44525 ms
template <typename T>
__global__ void sgemm_v1(T *out, const T *in, const T *weight, const T *bias,
                         size_t M, size_t N, size_t K) {
    int midx = blockIdx.y * blockDim.y + threadIdx.y;
    int nidx = blockIdx.x * blockDim.x + threadIdx.x;

    if (midx >= M || nidx >= N) {
        return;
    }

    float sum = 0.0f;
    if (bias != nullptr) {
        sum += to_float(bias[nidx]);
    }

    for (int k = 0; k < K; k++) {
        sum += to_float(in[midx * K + k]) * to_float(weight[nidx * K + k]);
    }

    out[midx * N + nidx] = from_float<T>(sum);
}

// v2：使用sharead memory实现，显著降低对global memory的访问次数实现加速
// v2_time:
// Torch time: 5.63606 ms
// WGINFER time: 43.84619 ms
// Torch time: 0.60475 ms
// WGINFER time: 49.69251 ms
// Torch time: 0.60049 ms
// WGINFER time: 50.35990 ms
template <typename T>
__global__ void sgemm_v2(T *out, const T *in, const T *weight, const T *bias,
                         size_t M, size_t N, size_t K) {
    constexpr int BM = 16;
    constexpr int BN = 16;
    constexpr int BK = 16;

    // NVIDIA GeForce GTX 4060 sharedMemPerBlock is 48KB = 48*1024B =
    // 49152B(0xc000) 1 float takes 4 Bytes, so (BM*BK + BK*BN) should <=
    // 48*1024/4 = 12288
    __shared__ float in_shared[BM * BK];
    __shared__ float weight_shared[BN * BK];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * BM + ty;
    int col = bx * BN + tx;

    float sum = 0.0f;
    if (bias != nullptr && col < N) {
        sum += to_float(bias[col]);
    }

    for (int k = 0; k < K; k += BK) {
        // 加载in：global memory -> shared memory
        if (row < M && (k + tx) < K) {
            in_shared[ty * BK + tx] = to_float(in[row * K + k + tx]);
        } else {
            in_shared[ty * BK + tx] = 0.0f;
        }

        // 加载weight
        if (col < N && (k + ty) < K) {
            weight_shared[tx * BK + ty] = to_float(weight[col * K + k + ty]);
        } else {
            weight_shared[tx * BK + ty] = 0.0f;
        }

        __syncthreads();

        // 在shared mem上进行当前bk的累加
        //// C[row, col] += sum_{k=0..BK-1} A[row, k+i] * W[col, k0+i]
        for (int i = 0; i < BK; i++) {
            sum += to_float(in_shared[ty * BK + i]) * to_float(weight_shared[tx * BK + i]);
        }
        __syncthreads();
    }

    if (by * BM + ty < M && bx * BN + tx < N) {
        out[row * N + col] = from_float<T>(sum);
    }
}

// v3：block tile 32x32 + thread tile 4x4，block 内 (8,8)=64 线程
// 每个线程计算一小块(4*4)，且数据复用加强，能显著增加计算强度
// v3_time:
// Torch time: 2.00178 ms
// WGINFER time: 20.16289 ms
// Torch time: 0.56751 ms
// WGINFER time: 20.26551 ms
// Torch time: 0.56799 ms
// WGINFER time: 20.25749 ms
template <typename T>
__global__ void sgemm_v3(T *out, const T *in, const T *weight, const T *bias,
                         size_t M, size_t N, size_t K) {
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int BK = 16;
    constexpr int TM = 4;
    constexpr int TN = 4;

    __shared__ float in_shared[BM * BK];
    __shared__ float weight_shared[BN * BK];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float sum[TM][TN];
    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            int col = bx * BN + tx * TN + j;
            sum[i][j] = (bias != nullptr && col < (int)N) ? to_float(bias[col]) : 0.0f;
        }
    }

    for (int k = 0; k < K; k += BK) {
        int tid = ty * blockDim.x + tx;
        int nthread = blockDim.x * blockDim.y;
        // 64 线程协作加载 in_shared[32][16]：每线程 8 个，coalesced
        for (int e = tid; e < BM * BK; e += nthread) {
            int r = e / BK;
            int c = e % BK;

            int global_r = by * BM + r;
            int global_c = k + c;

            in_shared[r * BK + c] = (global_r < M && global_c < K) ? to_float(in[global_r * K + global_c]) : 0.0f;
        }

        // load weight_shared[32][16]
        for (int e = tid; e < BN * BK; e += nthread) {
            int r = e / BK;
            int c = e % BK;

            int global_r = bx * BN + r;
            int global_c = k + c;

            weight_shared[r * BK + c] = (global_r < N && global_c < K) ? to_float(weight[global_r * K + global_c]) : 0.0f;
        }

        __syncthreads();

        // compute
        for (int kk = 0; kk < BK; kk++) {
            for (int i = 0; i < TM; i++) {
                float x = in_shared[(ty * TM + i) * BK + kk];
                for (int j = 0; j < TN; j++) {
                    sum[i][j] += x * weight_shared[(tx * TN + j) * BK + kk];
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            int row = by * BM + ty * TM + i;
            int col = bx * BN + tx * TN + j;
            if (row < (int)M && col < (int)N) {
                out[row * (int)N + col] = from_float<T>(sum[i][j]);
            }
        }
    }
}

// v4:将shared_mem上的数据搬运到reg上，计算时减少对shared_mem的访问
// v4_time:
// Torch time: 2.00347 ms
// WGINFER time: 14.46333 ms
// Torch time: 0.56831 ms
// WGINFER time: 14.59107 ms
// Torch time: 0.56920 ms
// WGINFER time: 14.59146 ms
template <typename T>
__global__ void sgemm_v4(T *out, const T *in, const T *weight, const T *bias,
                         size_t M, size_t N, size_t K) {
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int BK = 16;
    constexpr int TM = 4;
    constexpr int TN = 4;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    int block_row_base = by * BM;
    int block_col_base = bx * BN;
    int out_row_base = by * BM + ty * TM;
    int out_col_base = bx * BN + tx * TN;
    int nthread = blockDim.x * blockDim.y;

    __shared__ float in_shared[BM][BK];
    __shared__ float weight_shared[BN][BK];

    float sum[TM][TN] = {0.0f};
    float a_frag[TM];
    float b_frag[TN];

    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            sum[i][j] = (bias != nullptr && out_col_base + j < N) ? to_float(bias[out_col_base + j]) : 0.0f;
        }
    }

    for (int k = 0; k < K; k += BK) {
        // load in
        for (int i = tid; i < BM * BK; i += nthread) {
            int r = i / BK;
            int c = i % BK;
            in_shared[r][c] = ((block_row_base + r) < M && (k + c) < K) ? to_float(in[(block_row_base + r) * K + (k + c)]) : 0.0f;
        }

        // load weight
        for (int i = tid; i < BN * BK; i += nthread) {
            int r = i / BK;
            int c = i % BK;
            weight_shared[r][c] = ((block_col_base + r) < N && (k + c) < K) ? to_float(weight[(block_col_base + r) * K + (k + c)]) : 0.0f;
        }

        __syncthreads();

        for (int kk = 0; kk < BK; kk++) {
            // load：shared_mem to reg
            for (int i = 0; i < TM; i++) {
                a_frag[i] = in_shared[ty * TM + i][kk];
            }

            for (int j = 0; j < TN; j++) {
                b_frag[j] = weight_shared[tx * TN + j][kk];
            }

            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    sum[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            int r = by * BM + ty * TM + i;
            int c = bx * BN + tx * TN + j;
            if (r < (int)M && c < (int)N) {
                out[r * (int)N + c] = from_float<T>(sum[i][j]);
            }
        }
    }
}

// 1) global->shared 使用 float4 向量化加载
// 2) shared 中转置存储为 [BK, BM]/[BK, BN]，便于 thread-tile 连续读取
// 3) shared->register 用 float4 一次取 4 个元素，继续提高复用
// 4) 保留边界检查与尾块标量回退，保证通用输入尺寸正确
// Torch time: 2.01833 ms
// WGINFER time: 4.00644 ms
__global__ void sgemm_v5_float32(float *out, const float *in, const float *weight, const float *bias,
                                 size_t M, size_t N, size_t K) {
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int BK = 16;
    constexpr int TM = 4;
    constexpr int TN = 4;
    constexpr int VEC = 4;
    constexpr int BKV = CEIL(VEC, BK); // number of float4 groups along K in one BK-tile

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int nthread = blockDim.x * blockDim.y;

    const int block_row_base = by * BM;
    const int block_col_base = bx * BN;
    const int out_row_base = by * BM + ty * TM;
    const int out_col_base = bx * BN + tx * TN;

    __shared__ float As_t[BK][BM];
    __shared__ float Ws_t[BK][BN];

    float sum[TM][TN] = {0.0f};

    // Initialize accumulators with bias.
    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            const int out_c = out_col_base + j;
            sum[i][j] = (bias != nullptr && out_c < static_cast<int>(N)) ? bias[out_c] : 0.0f;
        }
    }

    for (int k = 0; k < K; k++) {
        // 1. prefetch
        for (int i = tid; i < BM * BKV; i += nthread) {
            const int r = i / BKV;
            const int vc = i % BKV;
            const int c = vc * VEC;
            const int gr = block_row_base + r;
            const int gc = k + c;

            float4 val{0};
            const size_t offset = gr * K + gc;
            if (gr < M) {
                if (gc + (VEC - 1) < K && (offset % VEC) == 0) {
                    val = LOAD_FLOAT4(in[offset]);
                } else {
                    if (gc < K) {
                        val.x = in[offset];
                    }
                    if (gc + 1 < K) {
                        val.y = in[offset + 1];
                    }
                    if (gc + 2 < K) {
                        val.z = in[offset + 2];
                    }
                    if (gc + 3 < K) {
                        val.w = in[offset + 3];
                    }
                }
            }
        }
    }
}

__global__ void sgemm_v5_half(half *out, const half *in, const half *weight, const half *bias,
                              size_t M, size_t N, size_t K) {
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int BK = 16;
    constexpr int TM = 4;
    constexpr int TN = 4;
}

__global__ void sgemm_v5_bfloat16(__nv_bfloat16 *out, const __nv_bfloat16 *in, const __nv_bfloat16 *weight, const __nv_bfloat16 *bias,
                                  size_t M, size_t N, size_t K) {
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int BK = 16;
    constexpr int TM = 4;
    constexpr int TN = 4;
}

// v6: 参考经典双缓冲 SGEMM 写法
// 1) global->shared 双缓冲
// 2) shared->register 使用 ping-pong frag，计算/取数流水化
template <const int BLOCK_SIZE_M,
          const int BLOCK_SIZE_N,
          const int BLOCK_SIZE_K,
          const int THREAD_SIZE_X,
          const int THREAD_SIZE_Y>
__global__ void sgemm_v6_float32(float *__restrict__ out,
                                 const float *__restrict__ in,
                                 const float *__restrict__ weight,
                                 const float *__restrict__ bias, size_t M,
                                 size_t N, size_t K) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int thread_x_per_block = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int thread_y_per_block = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int thread_num_per_block = thread_x_per_block * thread_y_per_block;

    const int tid = ty * thread_x_per_block + tx;

    __shared__ float As[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ float Bs[2][BLOCK_SIZE_K][BLOCK_SIZE_N];

    float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0.0f};

    float frag_a[2][THREAD_SIZE_Y];
    float frag_b[2][THREAD_SIZE_X];

    const int ldg_num_a = BLOCK_SIZE_M * BLOCK_SIZE_K / (thread_num_per_block * 4);
    const int ldg_num_b = BLOCK_SIZE_N * BLOCK_SIZE_K / (thread_num_per_block * 4);
    float ldg_a_reg[4 * ldg_num_a];
    float ldg_b_reg[4 * ldg_num_b];

    // A[M,K] and weight[N,K] are both contiguous along K.
    const int a_load_thread_per_row = BLOCK_SIZE_K / 4;
    const int b_load_thread_per_row = BLOCK_SIZE_K / 4;

    const int a_load_row_start = tid / a_load_thread_per_row;
    const int b_load_row_start = tid / b_load_thread_per_row;
    const int a_load_col = (tid % a_load_thread_per_row) * 4;
    const int b_load_col = (tid % b_load_thread_per_row) * 4;

    // 搬一行需要a_load_thread_per_row，总共有thread_num_per_block
    // 即能同时搬运的的行组数为thread_num_per_block / a_load_thread_per_row，下一次搬运则需要移动该组数
    const int a_load_row_stride = thread_num_per_block / a_load_thread_per_row;
    const int b_load_row_stride = thread_num_per_block / b_load_thread_per_row;

    const float *A = in + (BLOCK_SIZE_M * by) * K;
    const float *B = weight + (BLOCK_SIZE_N * bx) * K;

// prefetch first tile A: global -> registers -> shared
#pragma unroll
    for (int i = 0; i < BLOCK_SIZE_M; i += a_load_row_stride) {
        const int ldg_index = i / a_load_row_stride * 4;            // reg的起始索引
        const int offset = (a_load_row_start + i) * K + a_load_col; // 在global mem中的索引
        STORE_FLOAT4(ldg_a_reg[ldg_index]) = LOAD_FLOAT4(A[offset]);
        As[0][a_load_col][a_load_row_start + i] = ldg_a_reg[ldg_index];
        As[0][a_load_col + 1][a_load_row_start + i] = ldg_a_reg[ldg_index + 1];
        As[0][a_load_col + 2][a_load_row_start + i] = ldg_a_reg[ldg_index + 2];
        As[0][a_load_col + 3][a_load_row_start + i] = ldg_a_reg[ldg_index + 3];
    }

// prefetch first tile weight: global -> registers -> shared
#pragma unroll
    for (int i = 0; i < BLOCK_SIZE_N; i += b_load_row_stride) {
        const int ldg_index = i / b_load_row_stride * 4;
        const int offset = (b_load_row_start + i) * K + b_load_col;
        STORE_FLOAT4(ldg_b_reg[ldg_index]) = LOAD_FLOAT4(B[offset]);
        Bs[0][b_load_col][b_load_row_start + i] = ldg_b_reg[ldg_index];
        Bs[0][b_load_col + 1][b_load_row_start + i] = ldg_b_reg[ldg_index + 1];
        Bs[0][b_load_col + 2][b_load_row_start + i] = ldg_b_reg[ldg_index + 2];
        Bs[0][b_load_col + 3][b_load_row_start + i] = ldg_b_reg[ldg_index + 3];
    }
    __syncthreads();

// preload first k-slice from shared to registers
#pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
        STORE_FLOAT4(frag_a[0][thread_y]) = LOAD_FLOAT4(As[0][0][THREAD_SIZE_Y * ty + thread_y]);
    }
#pragma unroll
    for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
        STORE_FLOAT4(frag_b[0][thread_x]) = LOAD_FLOAT4(Bs[0][0][THREAD_SIZE_X * tx + thread_x]);
    }

    // write流向：global mem ---> ldg_reg ----> shared mem
    // read流向：shared mem ---> frag -----> accum ，指的是当前计算从哪个shared buffer读取
    int write_stage_idx = 1; // 写指针，下一块tile写到哪一块shared buffer
    int tile_idx = 0;        // 表示当前处理到K维度的哪个tile起点
    do {
        tile_idx += BLOCK_SIZE_K;

        // prefetch next tile from global to load_reg
        if (tile_idx < K) {
#pragma unroll
            for (int i = 0; i < BLOCK_SIZE_M; i += a_load_row_stride) {
                const int ldg_index = i / a_load_row_stride * 4;
                const int offset = (a_load_row_start + i) * K + (a_load_col + tile_idx);
                STORE_FLOAT4(ldg_a_reg[ldg_index]) = LOAD_FLOAT4(A[offset]);
            }
#pragma unroll
            for (int i = 0; i < BLOCK_SIZE_N; i += b_load_row_stride) {
                const int ldg_index = i / b_load_row_stride * 4;
                const int offset = (b_load_row_start + i) * K + (b_load_col + tile_idx);
                STORE_FLOAT4(ldg_b_reg[ldg_index]) = LOAD_FLOAT4(B[offset]);
            }
        }

        const int load_stage_idx = write_stage_idx ^ 1;

        // 同一个K-tile 内的double-buffer流水
        // 对于每个j做两个操作：预取下一片k(shared->reg)和计算当前片k(reg->fma)，二者交错进行，掩盖了从shared mem到reg传输延迟
        // 边界为block_size_k-1,因为每轮先加载j+1，最后一片会在循环外单独计算
#pragma unroll
        for (int j = 0; j < BLOCK_SIZE_K - 1; ++j) {
// preload next k-slice from shared
#pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
                STORE_FLOAT4(frag_a[(j + 1) % 2][thread_y]) = LOAD_FLOAT4(As[load_stage_idx][j + 1]
                                                                            [THREAD_SIZE_Y * ty + thread_y]);
            }
#pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
                STORE_FLOAT4(frag_b[(j + 1) % 2][thread_x]) = LOAD_FLOAT4(Bs[load_stage_idx][j + 1]
                                                                            [THREAD_SIZE_X * tx + thread_x]);
            }

// mma
#pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
#pragma unroll
                for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                    accum[thread_y][thread_x] += frag_a[j % 2][thread_y] * frag_b[j % 2][thread_x];
                }
            }
        }

        // commit prefetched global values from load_reg into shared
        if (tile_idx < K) {
#pragma unroll
            for (int i = 0; i < BLOCK_SIZE_M; i += a_load_row_stride) {
                const int ldg_index = i / a_load_row_stride * 4;
                As[write_stage_idx][a_load_col][a_load_row_start + i] = ldg_a_reg[ldg_index];
                As[write_stage_idx][a_load_col + 1][a_load_row_start + i] = ldg_a_reg[ldg_index + 1];
                As[write_stage_idx][a_load_col + 2][a_load_row_start + i] = ldg_a_reg[ldg_index + 2];
                As[write_stage_idx][a_load_col + 3][a_load_row_start + i] = ldg_a_reg[ldg_index + 3];
            }
#pragma unroll
            for (int i = 0; i < BLOCK_SIZE_N; i += b_load_row_stride) {
                const int ldg_index = i / b_load_row_stride * 4;
                Bs[write_stage_idx][b_load_col][b_load_row_start + i] = ldg_b_reg[ldg_index];
                Bs[write_stage_idx][b_load_col + 1][b_load_row_start + i] = ldg_b_reg[ldg_index + 1];
                Bs[write_stage_idx][b_load_col + 2][b_load_row_start + i] = ldg_b_reg[ldg_index + 2];
                Bs[write_stage_idx][b_load_col + 3][b_load_row_start + i] = ldg_b_reg[ldg_index + 3];
            }
            __syncthreads();
            write_stage_idx ^= 1;
        }

// compute last k-slice in current tile
// BK % 2 must == 0
#pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
            STORE_FLOAT4(frag_a[0][thread_y]) = LOAD_FLOAT4(As[load_stage_idx ^ 1][0]
                                                              [THREAD_SIZE_Y * ty + thread_y]);
        }
#pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
            STORE_FLOAT4(frag_b[0][thread_x]) = LOAD_FLOAT4(Bs[load_stage_idx ^ 1][0]
                                                              [THREAD_SIZE_X * tx + thread_x]);
        }
#pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
#pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                accum[thread_y][thread_x] += frag_a[1][thread_y] * frag_b[1][thread_x];
            }
        }
    } while (tile_idx < K);

    float bias_frag[THREAD_SIZE_X];
#pragma unroll
    for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
        const int col = BLOCK_SIZE_N * bx + tx * THREAD_SIZE_X + thread_x;
        bias_frag[thread_x] = (bias != nullptr) ? bias[col] : 0.0f;
    }

#pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
        const int row = BLOCK_SIZE_M * by + ty * THREAD_SIZE_Y + thread_y;
#pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
            const int col = BLOCK_SIZE_N * bx + tx * THREAD_SIZE_X + thread_x;
            float4 c_val;
            c_val.x = accum[thread_y][thread_x] + bias_frag[thread_x];
            c_val.y = accum[thread_y][thread_x + 1] + bias_frag[thread_x + 1];
            c_val.z = accum[thread_y][thread_x + 2] + bias_frag[thread_x + 2];
            c_val.w = accum[thread_y][thread_x + 3] + bias_frag[thread_x + 3];
            STORE_FLOAT4(out[row * N + col]) = c_val;
        }
    }
}

// v8: v6 的泛化版本，保留双缓冲主干并增加边界保护
template <const int BLOCK_SIZE_M,
          const int BLOCK_SIZE_N,
          const int BLOCK_SIZE_K,
          const int THREAD_SIZE_X,
          const int THREAD_SIZE_Y>
__global__ void sgemm_v8_float32(float *__restrict__ out,
                                 const float *__restrict__ in,
                                 const float *__restrict__ weight,
                                 const float *__restrict__ bias, size_t M,
                                 size_t N, size_t K) {
    static_assert(BLOCK_SIZE_K % 4 == 0, "BLOCK_SIZE_K must be a multiple of 4.");

    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int thread_x_per_block = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int thread_y_per_block = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int thread_num_per_block = thread_x_per_block * thread_y_per_block;

    const int tid = ty * thread_x_per_block + tx;

    __shared__ float As[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ float Bs[2][BLOCK_SIZE_K][BLOCK_SIZE_N];

    float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0.0f};
    float frag_a[2][THREAD_SIZE_Y];
    float frag_b[2][THREAD_SIZE_X];

    const int ldg_num_a = BLOCK_SIZE_M * BLOCK_SIZE_K / (thread_num_per_block * 4);
    const int ldg_num_b = BLOCK_SIZE_N * BLOCK_SIZE_K / (thread_num_per_block * 4);
    float ldg_a_reg[4 * ldg_num_a];
    float ldg_b_reg[4 * ldg_num_b];

    const int a_load_thread_per_row = BLOCK_SIZE_K / 4;
    const int b_load_thread_per_row = BLOCK_SIZE_K / 4;

    const int a_load_row_start = tid / a_load_thread_per_row;
    const int b_load_row_start = tid / b_load_thread_per_row;
    const int a_load_col = (tid % a_load_thread_per_row) * 4;
    const int b_load_col = (tid % b_load_thread_per_row) * 4;

    const int a_load_row_stride = thread_num_per_block / a_load_thread_per_row;
    const int b_load_row_stride = thread_num_per_block / b_load_thread_per_row;

    const float *A = in + (by * BLOCK_SIZE_M) * K;
    const float *B = weight + (bx * BLOCK_SIZE_N) * K;
    float *C = out + (by * BLOCK_SIZE_M) * N + (bx * BLOCK_SIZE_N);
    const float *bias_ptr = (bias != nullptr) ? (bias + bx * BLOCK_SIZE_N) : nullptr;

#pragma unroll
    for (int i = 0; i < BLOCK_SIZE_M; i += a_load_row_stride) {
        const int ldg_index = i / a_load_row_stride * 4;
        const size_t row = a_load_row_start + i;
        const size_t col = a_load_col;
        const bool row_in = by * BLOCK_SIZE_M + row < M;

        if (row_in && (col + 3) < K && is_aligned_16(&A[row * K + col])) {
            STORE_FLOAT4(ldg_a_reg[ldg_index]) = LOAD_FLOAT4(A[row * K + col]);
        } else {
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                const size_t c = col + v;
                ldg_a_reg[ldg_index + v] = (row_in && c < K) ? A[row * K + c] : 0.0f;
            }
        }

        As[0][a_load_col][a_load_row_start + i] = ldg_a_reg[ldg_index];
        As[0][a_load_col + 1][a_load_row_start + i] = ldg_a_reg[ldg_index + 1];
        As[0][a_load_col + 2][a_load_row_start + i] = ldg_a_reg[ldg_index + 2];
        As[0][a_load_col + 3][a_load_row_start + i] = ldg_a_reg[ldg_index + 3];
    }

#pragma unroll
    for (int i = 0; i < BLOCK_SIZE_N; i += b_load_row_stride) {
        const int ldg_index = i / b_load_row_stride * 4;
        const size_t row = b_load_row_start + i;
        const size_t col = b_load_col;
        const bool row_in = bx * BLOCK_SIZE_N + row < N;

        if (row_in && (col + 3) < K && is_aligned_16(&B[row * K + col])) {
            STORE_FLOAT4(ldg_b_reg[ldg_index]) = LOAD_FLOAT4(B[row * K + col]);
        } else {
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                const size_t c = col + static_cast<size_t>(v);
                ldg_b_reg[ldg_index + v] = (row_in && c < K) ? B[row * K + c] : 0.0f;
            }
        }

        Bs[0][b_load_col][b_load_row_start + i] = ldg_b_reg[ldg_index];
        Bs[0][b_load_col + 1][b_load_row_start + i] = ldg_b_reg[ldg_index + 1];
        Bs[0][b_load_col + 2][b_load_row_start + i] = ldg_b_reg[ldg_index + 2];
        Bs[0][b_load_col + 3][b_load_row_start + i] = ldg_b_reg[ldg_index + 3];
    }
    __syncthreads();

#pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
        STORE_FLOAT4(frag_a[0][thread_y]) = LOAD_FLOAT4(As[0][0][THREAD_SIZE_Y * ty + thread_y]);
    }
#pragma unroll
    for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
        STORE_FLOAT4(frag_b[0][thread_x]) = LOAD_FLOAT4(Bs[0][0][THREAD_SIZE_X * tx + thread_x]);
    }

    int write_stage_idx = 1;
    int tile_idx = 0;
    do {
        tile_idx += BLOCK_SIZE_K;

        if (tile_idx < K) {
#pragma unroll
            for (int i = 0; i < BLOCK_SIZE_M; i += a_load_row_stride) {
                const int ldg_index = i / a_load_row_stride * 4;
                const size_t row = a_load_row_start + i;
                const size_t col = a_load_col + tile_idx;
                const bool row_in = by * BLOCK_SIZE_M + row < M;

                if (row_in && (col + 3) < K && is_aligned_16(&A[row * K + col])) {
                    STORE_FLOAT4(ldg_a_reg[ldg_index]) = LOAD_FLOAT4(A[row * K + col]);
                } else {
#pragma unroll
                    for (int v = 0; v < 4; ++v) {
                        const size_t c = col + v;
                        ldg_a_reg[ldg_index + v] = (row_in && c < K) ? A[row * K + c] : 0.0f;
                    }
                }
            }

#pragma unroll
            for (int i = 0; i < BLOCK_SIZE_N; i += b_load_row_stride) {
                const int ldg_index = i / b_load_row_stride * 4;
                const size_t row = b_load_row_start + i;
                const size_t col = b_load_col + tile_idx;
                const bool row_in = bx * BLOCK_SIZE_N + row < N;

                if (row_in && (col + 3) < K && is_aligned_16(&B[row * K + col])) {
                    STORE_FLOAT4(ldg_b_reg[ldg_index]) = LOAD_FLOAT4(B[row * K + col]);
                } else {
#pragma unroll
                    for (int v = 0; v < 4; ++v) {
                        const size_t c = col + static_cast<size_t>(v);
                        ldg_b_reg[ldg_index + v] = (row_in && c < K) ? B[row * K + c] : 0.0f;
                    }
                }
            }
        }

        const int load_stage_idx = write_stage_idx ^ 1;

#pragma unroll
        for (int j = 0; j < BLOCK_SIZE_K - 1; ++j) {
#pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
                STORE_FLOAT4(frag_a[(j + 1) % 2][thread_y]) = LOAD_FLOAT4(As[load_stage_idx][j + 1][THREAD_SIZE_Y * ty + thread_y]);
            }
#pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
                STORE_FLOAT4(frag_b[(j + 1) % 2][thread_x]) = LOAD_FLOAT4(Bs[load_stage_idx][j + 1][THREAD_SIZE_X * tx + thread_x]);
            }

#pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
#pragma unroll
                for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                    accum[thread_y][thread_x] += frag_a[j % 2][thread_y] * frag_b[j % 2][thread_x];
                }
            }
        }

        if (tile_idx < static_cast<int>(K)) {
#pragma unroll
            for (int i = 0; i < BLOCK_SIZE_M; i += a_load_row_stride) {
                const int ldg_index = i / a_load_row_stride * 4;
                As[write_stage_idx][a_load_col][a_load_row_start + i] = ldg_a_reg[ldg_index];
                As[write_stage_idx][a_load_col + 1][a_load_row_start + i] = ldg_a_reg[ldg_index + 1];
                As[write_stage_idx][a_load_col + 2][a_load_row_start + i] = ldg_a_reg[ldg_index + 2];
                As[write_stage_idx][a_load_col + 3][a_load_row_start + i] = ldg_a_reg[ldg_index + 3];
            }
#pragma unroll
            for (int i = 0; i < BLOCK_SIZE_N; i += b_load_row_stride) {
                const int ldg_index = i / b_load_row_stride * 4;
                Bs[write_stage_idx][b_load_col][b_load_row_start + i] = ldg_b_reg[ldg_index];
                Bs[write_stage_idx][b_load_col + 1][b_load_row_start + i] = ldg_b_reg[ldg_index + 1];
                Bs[write_stage_idx][b_load_col + 2][b_load_row_start + i] = ldg_b_reg[ldg_index + 2];
                Bs[write_stage_idx][b_load_col + 3][b_load_row_start + i] = ldg_b_reg[ldg_index + 3];
            }
            __syncthreads();
            write_stage_idx ^= 1;
        }

#pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
            STORE_FLOAT4(frag_a[0][thread_y]) = LOAD_FLOAT4(As[load_stage_idx ^ 1][0]
                                                              [THREAD_SIZE_Y * ty + thread_y]);
        }
#pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
            STORE_FLOAT4(frag_b[0][thread_x]) = LOAD_FLOAT4(Bs[load_stage_idx ^ 1][0]
                                                              [THREAD_SIZE_X * tx + thread_x]);
        }
#pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
#pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                accum[thread_y][thread_x] += frag_a[1][thread_y] * frag_b[1][thread_x];
            }
        }
    } while (tile_idx < K);

    float bias_frag[THREAD_SIZE_X];
#pragma unroll
    for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
        const size_t col = tx * THREAD_SIZE_X + thread_x;
        const size_t global_col = bx * BLOCK_SIZE_N + col;
        bias_frag[thread_x] = (bias_ptr != nullptr && global_col < N) ? bias_ptr[col] : 0.0f;
    }

#pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
        const size_t row = ty * THREAD_SIZE_Y + thread_y;
        const size_t global_row = by * BLOCK_SIZE_M + row;
        if (global_row >= M) {
            continue;
        }
#pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
            const size_t col = tx * THREAD_SIZE_X + thread_x;
            const size_t global_col = bx * BLOCK_SIZE_N + col;
            float4 c_val;
            c_val.x = accum[thread_y][thread_x] + bias_frag[thread_x];
            c_val.y = accum[thread_y][thread_x + 1] + bias_frag[thread_x + 1];
            c_val.z = accum[thread_y][thread_x + 2] + bias_frag[thread_x + 2];
            c_val.w = accum[thread_y][thread_x + 3] + bias_frag[thread_x + 3];

            if ((global_col + 3) < N && is_aligned_16(&C[row * N + col])) {
                STORE_FLOAT4(C[row * N + col]) = c_val;
            } else {
                if (global_col < N) {
                    C[row * N + col] = c_val.x;
                }
                if (global_col + 1 < N) {
                    C[row * N + col + 1] = c_val.y;
                }
                if (global_col + 2 < N) {
                    C[row * N + col + 2] = c_val.z;
                }
                if (global_col + 3 < N) {
                    C[row * N + col + 3] = c_val.w;
                }
            }
        }
    }
}

template <const int BLOCK_SIZE_M,
          const int BLOCK_SIZE_N,
          const int BLOCK_SIZE_K,
          const int THREAD_SIZE_X,
          const int THREAD_SIZE_Y>
__global__ void sgemm_v6_half(half *__restrict__ out,
                              const half *__restrict__ in,
                              const half *__restrict__ weight,
                              const half *__restrict__ bias, size_t M,
                              size_t N, size_t K) {
    static_assert(BLOCK_SIZE_K % 4 == 0, "BLOCK_SIZE_K must be a multiple of 4.");
    static_assert(THREAD_SIZE_X % 2 == 0, "THREAD_SIZE_X must be even for half2 stores.");

    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int thread_x_per_block = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int thread_y_per_block = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int thread_num_per_block = thread_x_per_block * thread_y_per_block;

    const int tid = ty * thread_x_per_block + tx;

    __shared__ float As[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ float Bs[2][BLOCK_SIZE_K][BLOCK_SIZE_N];

    float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0.0f};

    float frag_a[2][THREAD_SIZE_Y];
    float frag_b[2][THREAD_SIZE_X];

    const int ldg_num_a = BLOCK_SIZE_M * BLOCK_SIZE_K / (thread_num_per_block * 4);
    const int ldg_num_b = BLOCK_SIZE_N * BLOCK_SIZE_K / (thread_num_per_block * 4);
    float ldg_a_reg[4 * ldg_num_a];
    float ldg_b_reg[4 * ldg_num_b];

    const int a_load_thread_per_row = BLOCK_SIZE_K / 4;
    const int b_load_thread_per_row = BLOCK_SIZE_K / 4;

    const int a_load_row_start = tid / a_load_thread_per_row;
    const int b_load_row_start = tid / b_load_thread_per_row;
    const int a_load_col = (tid % a_load_thread_per_row) * 4;
    const int b_load_col = (tid % b_load_thread_per_row) * 4;

    const int a_load_row_stride = thread_num_per_block / a_load_thread_per_row;
    const int b_load_row_stride = thread_num_per_block / b_load_thread_per_row;

    const half *A = in + (BLOCK_SIZE_M * by) * K;
    const half *B = weight + (BLOCK_SIZE_N * bx) * K;

#pragma unroll
    for (int i = 0; i < BLOCK_SIZE_M; i += a_load_row_stride) {
        const int ldg_index = i / a_load_row_stride * 4;
        const int offset = (a_load_row_start + i) * K + a_load_col;
        const half2 a_pack0 = LOAD_HALF2(A[offset]);
        const half2 a_pack1 = LOAD_HALF2(A[offset + 2]);
        const float2 a_f0 = __half22float2(a_pack0);
        const float2 a_f1 = __half22float2(a_pack1);
        ldg_a_reg[ldg_index] = a_f0.x;
        ldg_a_reg[ldg_index + 1] = a_f0.y;
        ldg_a_reg[ldg_index + 2] = a_f1.x;
        ldg_a_reg[ldg_index + 3] = a_f1.y;

        As[0][a_load_col][a_load_row_start + i] = ldg_a_reg[ldg_index];
        As[0][a_load_col + 1][a_load_row_start + i] = ldg_a_reg[ldg_index + 1];
        As[0][a_load_col + 2][a_load_row_start + i] = ldg_a_reg[ldg_index + 2];
        As[0][a_load_col + 3][a_load_row_start + i] = ldg_a_reg[ldg_index + 3];
    }

#pragma unroll
    for (int i = 0; i < BLOCK_SIZE_N; i += b_load_row_stride) {
        const int ldg_index = i / b_load_row_stride * 4;
        const int offset = (b_load_row_start + i) * K + b_load_col;
        const half2 b_pack0 = LOAD_HALF2(B[offset]);
        const half2 b_pack1 = LOAD_HALF2(B[offset + 2]);
        const float2 b_f0 = __half22float2(b_pack0);
        const float2 b_f1 = __half22float2(b_pack1);
        ldg_b_reg[ldg_index] = b_f0.x;
        ldg_b_reg[ldg_index + 1] = b_f0.y;
        ldg_b_reg[ldg_index + 2] = b_f1.x;
        ldg_b_reg[ldg_index + 3] = b_f1.y;

        Bs[0][b_load_col][b_load_row_start + i] = ldg_b_reg[ldg_index];
        Bs[0][b_load_col + 1][b_load_row_start + i] = ldg_b_reg[ldg_index + 1];
        Bs[0][b_load_col + 2][b_load_row_start + i] = ldg_b_reg[ldg_index + 2];
        Bs[0][b_load_col + 3][b_load_row_start + i] = ldg_b_reg[ldg_index + 3];
    }
    __syncthreads();

#pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
        STORE_FLOAT4(frag_a[0][thread_y]) = LOAD_FLOAT4(As[0][0][THREAD_SIZE_Y * ty + thread_y]);
    }
#pragma unroll
    for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
        STORE_FLOAT4(frag_b[0][thread_x]) = LOAD_FLOAT4(Bs[0][0][THREAD_SIZE_X * tx + thread_x]);
    }

    int write_stage_idx = 1;
    int tile_idx = 0;
    do {
        tile_idx += BLOCK_SIZE_K;

        if (tile_idx < K) {
#pragma unroll
            for (int i = 0; i < BLOCK_SIZE_M; i += a_load_row_stride) {
                const int ldg_index = i / a_load_row_stride * 4;
                const int offset = (a_load_row_start + i) * K + (a_load_col + tile_idx);
                const half2 a_pack0 = LOAD_HALF2(A[offset]);
                const half2 a_pack1 = LOAD_HALF2(A[offset + 2]);
                const float2 a_f0 = __half22float2(a_pack0);
                const float2 a_f1 = __half22float2(a_pack1);
                ldg_a_reg[ldg_index] = a_f0.x;
                ldg_a_reg[ldg_index + 1] = a_f0.y;
                ldg_a_reg[ldg_index + 2] = a_f1.x;
                ldg_a_reg[ldg_index + 3] = a_f1.y;
            }
#pragma unroll
            for (int i = 0; i < BLOCK_SIZE_N; i += b_load_row_stride) {
                const int ldg_index = i / b_load_row_stride * 4;
                const int offset = (b_load_row_start + i) * K + (b_load_col + tile_idx);
                const half2 b_pack0 = LOAD_HALF2(B[offset]);
                const half2 b_pack1 = LOAD_HALF2(B[offset + 2]);
                const float2 b_f0 = __half22float2(b_pack0);
                const float2 b_f1 = __half22float2(b_pack1);
                ldg_b_reg[ldg_index] = b_f0.x;
                ldg_b_reg[ldg_index + 1] = b_f0.y;
                ldg_b_reg[ldg_index + 2] = b_f1.x;
                ldg_b_reg[ldg_index + 3] = b_f1.y;
            }
        }

        const int load_stage_idx = write_stage_idx ^ 1;

#pragma unroll
        for (int j = 0; j < BLOCK_SIZE_K - 1; ++j) {
#pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
                STORE_FLOAT4(frag_a[(j + 1) % 2][thread_y]) = LOAD_FLOAT4(As[load_stage_idx][j + 1]
                                                                            [THREAD_SIZE_Y * ty + thread_y]);
            }
#pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
                STORE_FLOAT4(frag_b[(j + 1) % 2][thread_x]) = LOAD_FLOAT4(Bs[load_stage_idx][j + 1]
                                                                            [THREAD_SIZE_X * tx + thread_x]);
            }

#pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
#pragma unroll
                for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                    accum[thread_y][thread_x] += frag_a[j % 2][thread_y] * frag_b[j % 2][thread_x];
                }
            }
        }

        if (tile_idx < K) {
#pragma unroll
            for (int i = 0; i < BLOCK_SIZE_M; i += a_load_row_stride) {
                const int ldg_index = i / a_load_row_stride * 4;
                As[write_stage_idx][a_load_col][a_load_row_start + i] = ldg_a_reg[ldg_index];
                As[write_stage_idx][a_load_col + 1][a_load_row_start + i] = ldg_a_reg[ldg_index + 1];
                As[write_stage_idx][a_load_col + 2][a_load_row_start + i] = ldg_a_reg[ldg_index + 2];
                As[write_stage_idx][a_load_col + 3][a_load_row_start + i] = ldg_a_reg[ldg_index + 3];
            }
#pragma unroll
            for (int i = 0; i < BLOCK_SIZE_N; i += b_load_row_stride) {
                const int ldg_index = i / b_load_row_stride * 4;
                Bs[write_stage_idx][b_load_col][b_load_row_start + i] = ldg_b_reg[ldg_index];
                Bs[write_stage_idx][b_load_col + 1][b_load_row_start + i] = ldg_b_reg[ldg_index + 1];
                Bs[write_stage_idx][b_load_col + 2][b_load_row_start + i] = ldg_b_reg[ldg_index + 2];
                Bs[write_stage_idx][b_load_col + 3][b_load_row_start + i] = ldg_b_reg[ldg_index + 3];
            }
            __syncthreads();
            write_stage_idx ^= 1;
        }

#pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
            STORE_FLOAT4(frag_a[0][thread_y]) = LOAD_FLOAT4(As[load_stage_idx ^ 1][0]
                                                              [THREAD_SIZE_Y * ty + thread_y]);
        }
#pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
            STORE_FLOAT4(frag_b[0][thread_x]) = LOAD_FLOAT4(Bs[load_stage_idx ^ 1][0]
                                                              [THREAD_SIZE_X * tx + thread_x]);
        }
#pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
#pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                accum[thread_y][thread_x] += frag_a[1][thread_y] * frag_b[1][thread_x];
            }
        }
    } while (tile_idx < K);

    float bias_frag[THREAD_SIZE_X];
#pragma unroll
    for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 2) {
        const int col = BLOCK_SIZE_N * bx + tx * THREAD_SIZE_X + thread_x;
        if (bias != nullptr) {
            const half2 b_pack = LOAD_HALF2(bias[col]);
            const float2 b_f = __half22float2(b_pack);
            bias_frag[thread_x] = b_f.x;
            bias_frag[thread_x + 1] = b_f.y;
        } else {
            bias_frag[thread_x] = 0.0f;
            bias_frag[thread_x + 1] = 0.0f;
        }
    }

#pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
        const int row = BLOCK_SIZE_M * by + ty * THREAD_SIZE_Y + thread_y;
#pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 2) {
            const int col = BLOCK_SIZE_N * bx + tx * THREAD_SIZE_X + thread_x;
            const float out0 = accum[thread_y][thread_x] + bias_frag[thread_x];
            const float out1 = accum[thread_y][thread_x + 1] + bias_frag[thread_x + 1];
            STORE_HALF2(out[row * N + col]) = __floats2half2_rn(out0, out1);
        }
    }
}

template <const int BLOCK_SIZE_M,
          const int BLOCK_SIZE_N,
          const int BLOCK_SIZE_K,
          const int THREAD_SIZE_X,
          const int THREAD_SIZE_Y>
__global__ void sgemm_v6_bfloat16(__nv_bfloat16 *__restrict__ out,
                                  const __nv_bfloat16 *__restrict__ in,
                                  const __nv_bfloat16 *__restrict__ weight,
                                  const __nv_bfloat16 *__restrict__ bias,
                                  size_t M, size_t N, size_t K) {
    static_assert(BLOCK_SIZE_K % 4 == 0, "BLOCK_SIZE_K must be a multiple of 4.");
    static_assert(THREAD_SIZE_X % 2 == 0, "THREAD_SIZE_X must be even for bfloat162 stores.");

    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int thread_x_per_block = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int thread_y_per_block = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int thread_num_per_block = thread_x_per_block * thread_y_per_block;

    const int tid = ty * thread_x_per_block + tx;

    __shared__ float As[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ float Bs[2][BLOCK_SIZE_K][BLOCK_SIZE_N];

    float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0.0f};

    float frag_a[2][THREAD_SIZE_Y];
    float frag_b[2][THREAD_SIZE_X];

    const int ldg_num_a = BLOCK_SIZE_M * BLOCK_SIZE_K / (thread_num_per_block * 4);
    const int ldg_num_b = BLOCK_SIZE_N * BLOCK_SIZE_K / (thread_num_per_block * 4);
    float ldg_a_reg[4 * ldg_num_a];
    float ldg_b_reg[4 * ldg_num_b];

    const int a_load_thread_per_row = BLOCK_SIZE_K / 4;
    const int b_load_thread_per_row = BLOCK_SIZE_K / 4;

    const int a_load_row_start = tid / a_load_thread_per_row;
    const int b_load_row_start = tid / b_load_thread_per_row;
    const int a_load_col = (tid % a_load_thread_per_row) * 4;
    const int b_load_col = (tid % b_load_thread_per_row) * 4;

    const int a_load_row_stride = thread_num_per_block / a_load_thread_per_row;
    const int b_load_row_stride = thread_num_per_block / b_load_thread_per_row;

    const __nv_bfloat16 *A = in + (BLOCK_SIZE_M * by) * K;
    const __nv_bfloat16 *B = weight + (BLOCK_SIZE_N * bx) * K;

#pragma unroll
    for (int i = 0; i < BLOCK_SIZE_M; i += a_load_row_stride) {
        const int ldg_index = i / a_load_row_stride * 4;
        const int offset = (a_load_row_start + i) * K + a_load_col;
        const __nv_bfloat162 a_pack0 = LOAD_BFLOAT2(A[offset]);
        const __nv_bfloat162 a_pack1 = LOAD_BFLOAT2(A[offset + 2]);
        const float2 a_f0 = __bfloat1622float2(a_pack0);
        const float2 a_f1 = __bfloat1622float2(a_pack1);
        ldg_a_reg[ldg_index] = a_f0.x;
        ldg_a_reg[ldg_index + 1] = a_f0.y;
        ldg_a_reg[ldg_index + 2] = a_f1.x;
        ldg_a_reg[ldg_index + 3] = a_f1.y;

        As[0][a_load_col][a_load_row_start + i] = ldg_a_reg[ldg_index];
        As[0][a_load_col + 1][a_load_row_start + i] = ldg_a_reg[ldg_index + 1];
        As[0][a_load_col + 2][a_load_row_start + i] = ldg_a_reg[ldg_index + 2];
        As[0][a_load_col + 3][a_load_row_start + i] = ldg_a_reg[ldg_index + 3];
    }

#pragma unroll
    for (int i = 0; i < BLOCK_SIZE_N; i += b_load_row_stride) {
        const int ldg_index = i / b_load_row_stride * 4;
        const int offset = (b_load_row_start + i) * K + b_load_col;
        const __nv_bfloat162 b_pack0 = LOAD_BFLOAT2(B[offset]);
        const __nv_bfloat162 b_pack1 = LOAD_BFLOAT2(B[offset + 2]);
        const float2 b_f0 = __bfloat1622float2(b_pack0);
        const float2 b_f1 = __bfloat1622float2(b_pack1);
        ldg_b_reg[ldg_index] = b_f0.x;
        ldg_b_reg[ldg_index + 1] = b_f0.y;
        ldg_b_reg[ldg_index + 2] = b_f1.x;
        ldg_b_reg[ldg_index + 3] = b_f1.y;

        Bs[0][b_load_col][b_load_row_start + i] = ldg_b_reg[ldg_index];
        Bs[0][b_load_col + 1][b_load_row_start + i] = ldg_b_reg[ldg_index + 1];
        Bs[0][b_load_col + 2][b_load_row_start + i] = ldg_b_reg[ldg_index + 2];
        Bs[0][b_load_col + 3][b_load_row_start + i] = ldg_b_reg[ldg_index + 3];
    }
    __syncthreads();

#pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
        STORE_FLOAT4(frag_a[0][thread_y]) = LOAD_FLOAT4(As[0][0][THREAD_SIZE_Y * ty + thread_y]);
    }
#pragma unroll
    for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
        STORE_FLOAT4(frag_b[0][thread_x]) = LOAD_FLOAT4(Bs[0][0][THREAD_SIZE_X * tx + thread_x]);
    }

    int write_stage_idx = 1;
    int tile_idx = 0;
    do {
        tile_idx += BLOCK_SIZE_K;

        if (tile_idx < K) {
#pragma unroll
            for (int i = 0; i < BLOCK_SIZE_M; i += a_load_row_stride) {
                const int ldg_index = i / a_load_row_stride * 4;
                const int offset = (a_load_row_start + i) * K + (a_load_col + tile_idx);
                const __nv_bfloat162 a_pack0 = LOAD_BFLOAT2(A[offset]);
                const __nv_bfloat162 a_pack1 = LOAD_BFLOAT2(A[offset + 2]);
                const float2 a_f0 = __bfloat1622float2(a_pack0);
                const float2 a_f1 = __bfloat1622float2(a_pack1);
                ldg_a_reg[ldg_index] = a_f0.x;
                ldg_a_reg[ldg_index + 1] = a_f0.y;
                ldg_a_reg[ldg_index + 2] = a_f1.x;
                ldg_a_reg[ldg_index + 3] = a_f1.y;
            }
#pragma unroll
            for (int i = 0; i < BLOCK_SIZE_N; i += b_load_row_stride) {
                const int ldg_index = i / b_load_row_stride * 4;
                const int offset = (b_load_row_start + i) * K + (b_load_col + tile_idx);
                const __nv_bfloat162 b_pack0 = LOAD_BFLOAT2(B[offset]);
                const __nv_bfloat162 b_pack1 = LOAD_BFLOAT2(B[offset + 2]);
                const float2 b_f0 = __bfloat1622float2(b_pack0);
                const float2 b_f1 = __bfloat1622float2(b_pack1);
                ldg_b_reg[ldg_index] = b_f0.x;
                ldg_b_reg[ldg_index + 1] = b_f0.y;
                ldg_b_reg[ldg_index + 2] = b_f1.x;
                ldg_b_reg[ldg_index + 3] = b_f1.y;
            }
        }

        const int load_stage_idx = write_stage_idx ^ 1;

#pragma unroll
        for (int j = 0; j < BLOCK_SIZE_K - 1; ++j) {
#pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
                STORE_FLOAT4(frag_a[(j + 1) % 2][thread_y]) = LOAD_FLOAT4(As[load_stage_idx][j + 1]
                                                                            [THREAD_SIZE_Y * ty + thread_y]);
            }
#pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
                STORE_FLOAT4(frag_b[(j + 1) % 2][thread_x]) = LOAD_FLOAT4(Bs[load_stage_idx][j + 1]
                                                                            [THREAD_SIZE_X * tx + thread_x]);
            }

#pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
#pragma unroll
                for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                    accum[thread_y][thread_x] += frag_a[j % 2][thread_y] * frag_b[j % 2][thread_x];
                }
            }
        }

        if (tile_idx < K) {
#pragma unroll
            for (int i = 0; i < BLOCK_SIZE_M; i += a_load_row_stride) {
                const int ldg_index = i / a_load_row_stride * 4;
                As[write_stage_idx][a_load_col][a_load_row_start + i] = ldg_a_reg[ldg_index];
                As[write_stage_idx][a_load_col + 1][a_load_row_start + i] = ldg_a_reg[ldg_index + 1];
                As[write_stage_idx][a_load_col + 2][a_load_row_start + i] = ldg_a_reg[ldg_index + 2];
                As[write_stage_idx][a_load_col + 3][a_load_row_start + i] = ldg_a_reg[ldg_index + 3];
            }
#pragma unroll
            for (int i = 0; i < BLOCK_SIZE_N; i += b_load_row_stride) {
                const int ldg_index = i / b_load_row_stride * 4;
                Bs[write_stage_idx][b_load_col][b_load_row_start + i] = ldg_b_reg[ldg_index];
                Bs[write_stage_idx][b_load_col + 1][b_load_row_start + i] = ldg_b_reg[ldg_index + 1];
                Bs[write_stage_idx][b_load_col + 2][b_load_row_start + i] = ldg_b_reg[ldg_index + 2];
                Bs[write_stage_idx][b_load_col + 3][b_load_row_start + i] = ldg_b_reg[ldg_index + 3];
            }
            __syncthreads();
            write_stage_idx ^= 1;
        }

#pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
            STORE_FLOAT4(frag_a[0][thread_y]) = LOAD_FLOAT4(As[load_stage_idx ^ 1][0]
                                                              [THREAD_SIZE_Y * ty + thread_y]);
        }
#pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
            STORE_FLOAT4(frag_b[0][thread_x]) = LOAD_FLOAT4(Bs[load_stage_idx ^ 1][0]
                                                              [THREAD_SIZE_X * tx + thread_x]);
        }
#pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
#pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                accum[thread_y][thread_x] += frag_a[1][thread_y] * frag_b[1][thread_x];
            }
        }
    } while (tile_idx < K);

    float bias_frag[THREAD_SIZE_X];
#pragma unroll
    for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 2) {
        const int col = BLOCK_SIZE_N * bx + tx * THREAD_SIZE_X + thread_x;
        if (bias != nullptr) {
            const __nv_bfloat162 b_pack = LOAD_BFLOAT2(bias[col]);
            const float2 b_f = __bfloat1622float2(b_pack);
            bias_frag[thread_x] = b_f.x;
            bias_frag[thread_x + 1] = b_f.y;
        } else {
            bias_frag[thread_x] = 0.0f;
            bias_frag[thread_x + 1] = 0.0f;
        }
    }

#pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
        const int row = BLOCK_SIZE_M * by + ty * THREAD_SIZE_Y + thread_y;
#pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 2) {
            const int col = BLOCK_SIZE_N * bx + tx * THREAD_SIZE_X + thread_x;
            const float out0 = accum[thread_y][thread_x] + bias_frag[thread_x];
            const float out1 = accum[thread_y][thread_x + 1] + bias_frag[thread_x + 1];
            STORE_BFLOAT2(out[row * N + col]) = __floats2bfloat162_rn(out0, out1);
        }
    }
}

template <const int BLOCK_SIZE_M,
          const int BLOCK_SIZE_N,
          const int BLOCK_SIZE_K,
          const int THREAD_SIZE_X,
          const int THREAD_SIZE_Y>
__global__ void sgemm_v7_float32(float *__restrict__ out,
                                 const float *__restrict__ in,
                                 const float *__restrict__ weight,
                                 const float *__restrict__ bias, size_t M,
                                 size_t N, size_t K) {
    static_assert(BLOCK_SIZE_M == 128 && BLOCK_SIZE_N == 128 && BLOCK_SIZE_K == 8 && THREAD_SIZE_X == 8 && THREAD_SIZE_Y == 8,
                  "v7 is tuned for 128x128x8 tile and 8x8 thread tile.");

    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int thread_x_per_block = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int thread_y_per_block = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int thread_num_per_block = thread_x_per_block * thread_y_per_block;

    const int tid = ty * thread_x_per_block + tx;

    __shared__ float As[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ float Bs[2][BLOCK_SIZE_K][BLOCK_SIZE_N];

    float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0.0f};

    float frag_a[2][THREAD_SIZE_Y];
    float frag_b[2][THREAD_SIZE_X];

    const int ldg_num_a = BLOCK_SIZE_M * BLOCK_SIZE_K / (thread_num_per_block * 4);
    const int ldg_num_b = BLOCK_SIZE_K * BLOCK_SIZE_N / (thread_num_per_block * 4);
    float ldg_a_reg[4 * ldg_num_a];
    float ldg_b_reg[4 * ldg_num_b];

    // A and weight are row-major [M,K] / [N,K], so load weight across K.
    const int a_load_thread_per_row = BLOCK_SIZE_K / 4;
    const int b_load_thread_per_row = BLOCK_SIZE_K / 4;

    const int a_load_row_start = tid / a_load_thread_per_row;
    const int b_load_row_start = tid / b_load_thread_per_row;
    const int a_load_col = (tid % a_load_thread_per_row) * 4;
    const int b_load_col = (tid % b_load_thread_per_row) * 4;

    const int a_load_row_stride = thread_num_per_block / a_load_thread_per_row;
    const int b_load_row_stride = thread_num_per_block / b_load_thread_per_row;

    const float *A = &in[(BLOCK_SIZE_M * by) * K];
    const float *B = &weight[(BLOCK_SIZE_N * bx) * K];

// transfer first tile from global to shared
#pragma unroll
    for (int i = 0; i < BLOCK_SIZE_M; i += a_load_row_stride) {
        const int ldg_index = i / a_load_row_stride * 4;
        const int offset = (a_load_row_start + i) * K + a_load_col;
        STORE_FLOAT4(ldg_a_reg[ldg_index]) = LOAD_FLOAT4(A[offset]);
        As[0][a_load_col][a_load_row_start + i] = ldg_a_reg[ldg_index];
        As[0][a_load_col + 1][a_load_row_start + i] = ldg_a_reg[ldg_index + 1];
        As[0][a_load_col + 2][a_load_row_start + i] = ldg_a_reg[ldg_index + 2];
        As[0][a_load_col + 3][a_load_row_start + i] = ldg_a_reg[ldg_index + 3];
    }

#pragma unroll
    for (int i = 0; i < BLOCK_SIZE_N; i += b_load_row_stride) {
        const int ldg_index = i / b_load_row_stride * 4;
        const int offset = (b_load_row_start + i) * K + b_load_col;
        STORE_FLOAT4(ldg_b_reg[ldg_index]) = LOAD_FLOAT4(B[offset]);
        Bs[0][b_load_col][b_load_row_start + i] = ldg_b_reg[ldg_index];
        Bs[0][b_load_col + 1][b_load_row_start + i] = ldg_b_reg[ldg_index + 1];
        Bs[0][b_load_col + 2][b_load_row_start + i] = ldg_b_reg[ldg_index + 2];
        Bs[0][b_load_col + 3][b_load_row_start + i] = ldg_b_reg[ldg_index + 3];
    }
    __syncthreads();

    // load index of the tile (warp-level mapping)
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int a_tile_index = warp_id / 2 * 16 + lane_id / 8 * 4;
    const int b_tile_index = warp_id % 2 * 32 + lane_id % 8 * 4;

    // first slice: shared -> registers
    STORE_FLOAT4(frag_a[0][0]) = LOAD_FLOAT4(As[0][0][a_tile_index]);
    STORE_FLOAT4(frag_a[0][4]) = LOAD_FLOAT4(As[0][0][a_tile_index + BLOCK_SIZE_M / 2]);
    STORE_FLOAT4(frag_b[0][0]) = LOAD_FLOAT4(Bs[0][0][b_tile_index]);
    STORE_FLOAT4(frag_b[0][4]) = LOAD_FLOAT4(Bs[0][0][b_tile_index + BLOCK_SIZE_N / 2]);

    int write_stage_idx = 1;
    int tile_idx = 0;
    do {
        tile_idx += BLOCK_SIZE_K;
        if (tile_idx < K) {
#pragma unroll
            for (int i = 0; i < BLOCK_SIZE_M; i += a_load_row_stride) {
                const int ldg_index = i / a_load_row_stride * 4;
                const int offset = (a_load_row_start + i) * K + (a_load_col + tile_idx);
                STORE_FLOAT4(ldg_a_reg[ldg_index]) = LOAD_FLOAT4(A[offset]);
            }
#pragma unroll
            for (int i = 0; i < BLOCK_SIZE_N; i += b_load_row_stride) {
                const int ldg_index = i / b_load_row_stride * 4;
                const int offset = (b_load_row_start + i) * K + (b_load_col + tile_idx);
                STORE_FLOAT4(ldg_b_reg[ldg_index]) = LOAD_FLOAT4(B[offset]);
            }
        }

        const int load_stage_idx = write_stage_idx ^ 1;

#pragma unroll
        for (int j = 0; j < BLOCK_SIZE_K - 1; ++j) {
            STORE_FLOAT4(frag_a[(j + 1) % 2][0]) = LOAD_FLOAT4(As[load_stage_idx][j + 1][a_tile_index]);
            STORE_FLOAT4(frag_a[(j + 1) % 2][4]) = LOAD_FLOAT4(As[load_stage_idx][j + 1]
                                                                 [a_tile_index + BLOCK_SIZE_M / 2]);
            STORE_FLOAT4(frag_b[(j + 1) % 2][0]) = LOAD_FLOAT4(Bs[load_stage_idx][j + 1][b_tile_index]);
            STORE_FLOAT4(frag_b[(j + 1) % 2][4]) = LOAD_FLOAT4(Bs[load_stage_idx][j + 1]
                                                                 [b_tile_index + BLOCK_SIZE_N / 2]);

#pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
#pragma unroll
                for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                    accum[thread_y][thread_x] += frag_a[j % 2][thread_y] * frag_b[j % 2][thread_x];
                }
            }
        }

        if (tile_idx < K) {
#pragma unroll
            for (int i = 0; i < BLOCK_SIZE_M; i += a_load_row_stride) {
                const int ldg_index = i / a_load_row_stride * 4;
                As[write_stage_idx][a_load_col][a_load_row_start + i] = ldg_a_reg[ldg_index];
                As[write_stage_idx][a_load_col + 1][a_load_row_start + i] = ldg_a_reg[ldg_index + 1];
                As[write_stage_idx][a_load_col + 2][a_load_row_start + i] = ldg_a_reg[ldg_index + 2];
                As[write_stage_idx][a_load_col + 3][a_load_row_start + i] = ldg_a_reg[ldg_index + 3];
            }
#pragma unroll
            for (int i = 0; i < BLOCK_SIZE_N; i += b_load_row_stride) {
                const int ldg_index = i / b_load_row_stride * 4;
                Bs[write_stage_idx][b_load_col][b_load_row_start + i] = ldg_b_reg[ldg_index];
                Bs[write_stage_idx][b_load_col + 1][b_load_row_start + i] = ldg_b_reg[ldg_index + 1];
                Bs[write_stage_idx][b_load_col + 2][b_load_row_start + i] = ldg_b_reg[ldg_index + 2];
                Bs[write_stage_idx][b_load_col + 3][b_load_row_start + i] = ldg_b_reg[ldg_index + 3];
            }
            __syncthreads();
            write_stage_idx ^= 1;
        }

        STORE_FLOAT4(frag_a[0][0]) = LOAD_FLOAT4(As[load_stage_idx ^ 1][0][a_tile_index]);
        STORE_FLOAT4(frag_a[0][4]) = LOAD_FLOAT4(As[load_stage_idx ^ 1][0][a_tile_index + BLOCK_SIZE_M / 2]);
        STORE_FLOAT4(frag_b[0][0]) = LOAD_FLOAT4(Bs[load_stage_idx ^ 1][0][b_tile_index]);
        STORE_FLOAT4(frag_b[0][4]) = LOAD_FLOAT4(Bs[load_stage_idx ^ 1][0][b_tile_index + BLOCK_SIZE_N / 2]);

#pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
#pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                accum[thread_y][thread_x] += frag_a[1][thread_y] * frag_b[1][thread_x];
            }
        }
    } while (tile_idx < K);

    const int c_block_row = a_tile_index;
    const int c_block_col = b_tile_index;

    // store C00 block
    for (int i = 0; i < 4; i++) {
        const int row = BLOCK_SIZE_M * by + c_block_row + i;
        const int col = BLOCK_SIZE_N * bx + c_block_col;
        float4 c_val;
        c_val.x = accum[i][0];
        c_val.y = accum[i][1];
        c_val.z = accum[i][2];
        c_val.w = accum[i][3];
        if (bias != nullptr) {
            c_val.x += bias[col];
            c_val.y += bias[col + 1];
            c_val.z += bias[col + 2];
            c_val.w += bias[col + 3];
        }
        STORE_FLOAT4(out[row * N + col]) = c_val;
    }
    // store C01 block
    for (int i = 0; i < 4; i++) {
        const int row = BLOCK_SIZE_M * by + c_block_row + i;
        const int col = BLOCK_SIZE_N * bx + c_block_col + BLOCK_SIZE_N / 2;
        float4 c_val;
        c_val.x = accum[i][4];
        c_val.y = accum[i][5];
        c_val.z = accum[i][6];
        c_val.w = accum[i][7];
        if (bias != nullptr) {
            c_val.x += bias[col];
            c_val.y += bias[col + 1];
            c_val.z += bias[col + 2];
            c_val.w += bias[col + 3];
        }
        STORE_FLOAT4(out[row * N + col]) = c_val;
    }
    // store C10 block
    for (int i = 0; i < 4; i++) {
        const int row = BLOCK_SIZE_M * by + c_block_row + BLOCK_SIZE_M / 2 + i;
        const int col = BLOCK_SIZE_N * bx + c_block_col;
        float4 c_val;
        c_val.x = accum[i + 4][0];
        c_val.y = accum[i + 4][1];
        c_val.z = accum[i + 4][2];
        c_val.w = accum[i + 4][3];
        if (bias != nullptr) {
            c_val.x += bias[col];
            c_val.y += bias[col + 1];
            c_val.z += bias[col + 2];
            c_val.w += bias[col + 3];
        }
        STORE_FLOAT4(out[row * N + col]) = c_val;
    }
    // store C11 block
    for (int i = 0; i < 4; i++) {
        const int row = BLOCK_SIZE_M * by + c_block_row + BLOCK_SIZE_M / 2 + i;
        const int col = BLOCK_SIZE_N * bx + c_block_col + BLOCK_SIZE_N / 2;
        float4 c_val;
        c_val.x = accum[i + 4][4];
        c_val.y = accum[i + 4][5];
        c_val.z = accum[i + 4][6];
        c_val.w = accum[i + 4][7];
        if (bias != nullptr) {
            c_val.x += bias[col];
            c_val.y += bias[col + 1];
            c_val.z += bias[col + 2];
            c_val.w += bias[col + 3];
        }
        STORE_FLOAT4(out[row * N + col]) = c_val;
    }
}

} // namespace

namespace wginfer::ops::nvidia {
void linear(std::byte *out, const std::byte *in, const std::byte *weight,
            const std::byte *bias, wginferDataType_t type, size_t M, size_t N,
            size_t K) {
    switch (type) {
    case WGINFER_DTYPE_F32:
        linear_cublas_f32(reinterpret_cast<float *>(out),
                          reinterpret_cast<const float *>(in),
                          reinterpret_cast<const float *>(weight),
                          reinterpret_cast<const float *>(bias), M, N, K);
        break;
    case WGINFER_DTYPE_F16:
        linear_cublas_f16(reinterpret_cast<half *>(out),
                          reinterpret_cast<const half *>(in),
                          reinterpret_cast<const half *>(weight),
                          reinterpret_cast<const half *>(bias), M, N, K);
        break;
    case WGINFER_DTYPE_BF16:
        linear_cublas_bf16(reinterpret_cast<__nv_bfloat16 *>(out),
                           reinterpret_cast<const __nv_bfloat16 *>(in),
                           reinterpret_cast<const __nv_bfloat16 *>(weight),
                           reinterpret_cast<const __nv_bfloat16 *>(bias), M, N,
                           K);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    CUDA_CHECK(cudaGetLastError());
}
} // namespace wginfer::ops::nvidia
