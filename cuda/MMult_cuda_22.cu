#include <assert.h>
#include <stdlib.h>

// CUDA runtime
#include "helper.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

/**
 * SM 实现
 */
template <int BLOCK>
__global__ void sgemm(int m, int n, int k, float *a, int lda, float *b, int ldb,
                      float *c, int ldc) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    float * begin_a = a + bx * BLOCK * lda;
    float * begin_b = b + by * BLOCK;
    float * end_d = begin_a + lda;

    float sum = 0;
    for (float * ptr_a = begin_a, * ptr_b = begin_b; ptr_a < end_d; ptr_a += BLOCK, ptr_b += BLOCK * ldb) {
        __shared__ float s_a[BLOCK][BLOCK];
        __shared__ float s_b[BLOCK][BLOCK];
        s_a[ty][tx] = ptr_a[ty * lda + tx];
        s_b[ty][tx] = ptr_b[ty * ldb + tx];
        __syncthreads();

#pragma unroll
        for (int i = 0; i < BLOCK; i++) {
            sum += s_a[ty][i] * s_b[i][tx];
        }
        __syncthreads();
    }
    c[(bx * BLOCK + ty) * ldc + by * BLOCK + tx] = sum;
}

// template <int BLOCK>
// __global__ void sgemm(int m, int n, int k, float *a, int lda, float *b, int ldb,
//                       float *c, int ldc) {
//     int tx = threadIdx.x;
//     int ty = threadIdx.y;
//     int bx = blockIdx.x;
//     int by = blockIdx.y;
//     float * begin_a = a + bx * BLOCK * lda;
//     float * begin_b = b + by * BLOCK;
//     float * end_d = begin_a + lda;

//     float sum = 0;
//     for (float * ptr_a = begin_a, * ptr_b = begin_b; ptr_a < end_d; ptr_a += BLOCK, ptr_b += BLOCK * ldb) {
//         __shared__ float s_a[BLOCK][BLOCK];
//         __shared__ float s_b[BLOCK][BLOCK];
//         s_a[tx][ty] = ptr_a[tx * lda + ty];
//         s_b[tx][ty] = ptr_b[tx * ldb + ty];
//         __syncthreads();

// #pragma unroll
//         for (int i = 0; i < BLOCK; i++) {
//             sum += s_a[tx][i] * s_b[i][ty];
//         }
//         __syncthreads();
//     }
//     c[(bx * BLOCK + tx) * ldc + by * BLOCK + ty] = sum;
// }

void MY_MMult(cublasHandle_t handle, int m, int n, int k, float *d_A, int lda,
              float *d_B, int ldb, float *d_C, int ldc) {

  constexpr int BLOCK = 16;
  // subm, subn, subk
  dim3 block(BLOCK, BLOCK);
  dim3 grid((m + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);

  sgemm<BLOCK><<<grid, block>>>(m, n, k, d_A, lda, d_B, ldb, d_C, ldc);
}
