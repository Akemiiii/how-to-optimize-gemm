#include <assert.h>
#include <stdlib.h>

// CUDA runtime
#include "helper.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

/**
 * SM 实现
 */
template <int BLOCK, int STRIDE>
__global__ void sgemm(int m, int n, int k, float *a, int lda, float *b, int ldb,
                      float *c, int ldc) {
    constexpr int STEP = BLOCK * STRIDE;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    float * begin_a = a + bx * STEP * lda;
    float * begin_b = b + by * STEP;
    float * end_d = begin_a + lda;

    float sum[STRIDE][STRIDE] = {0.0};
    for (float * ptr_a = begin_a, * ptr_b = begin_b; ptr_a < end_d; ptr_a += STEP, ptr_b += STEP * ldb) {
        __shared__ float s_a[STEP][STEP];
        __shared__ float s_b[STEP][STEP];

#pragma unroll
        for (int i = 0; i < STRIDE; i++) {
#pragma unroll
            for (int j = 0; j < STRIDE; j++) {
                s_a[ty + j * STRIDE][tx + i * STRIDE] = ptr_a[(ty + j * STRIDE) * lda + tx + i * STRIDE];
                s_b[ty + j * STRIDE][tx + i * STRIDE] = ptr_b[(ty + j * STRIDE) * ldb + tx + i * STRIDE];
            }
        }
        __syncthreads();

#pragma unroll
        for (int i = 0; i < BLOCK; i++) {
            sum += s_a[ty][i] * s_b[i][tx];
        }
        __syncthreads();
    }
    c[(bx * BLOCK + ty) * ldc + by * BLOCK + tx] = sum;
}

void MY_MMult(cublasHandle_t handle, int m, int n, int k, float *d_A, int lda,
              float *d_B, int ldb, float *d_C, int ldc) {

  constexpr int BLOCK = 16;
  constexpr int STRIDE = 4;
  // subm, subn, subk
  dim3 block(BLOCK, BLOCK);
  dim3 grid((m + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);

  sgemm<BLOCK, STRIDE><<<grid, block>>>(m, n, k, d_A, lda, d_B, ldb, d_C, ldc);
}
