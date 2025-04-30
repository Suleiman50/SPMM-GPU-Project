
#include "common.h"
#include <stddef.h> 
#include <stdio.h>

__global__ __launch_bounds__(1024)
static void spmspm_gpu1_kernel(
    unsigned int rowsA,
    const unsigned int *A_rowPtrs,
    const unsigned int *A_colIdxs,
    const float *A_vals,
    const unsigned int *B_rowPtrs,
    const unsigned int *B_colIdxs,
    const float *B_vals,
    unsigned int colsB,
    unsigned int *C_rowIdxs,
    unsigned int *C_colIdxs,
    float *C_vals,
    int *marker,
    unsigned int *pos_out)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rowsA) return;

    int *m = marker + i * colsB;
#pragma unroll
    for (unsigned int j = 0; j < colsB; ++j)
        m[j] = -1;

    for (unsigned int pa = A_rowPtrs[i]; pa < A_rowPtrs[i + 1]; ++pa) {
        unsigned int k = A_colIdxs[pa];
        float vA = A_vals[pa];

        for (unsigned int pb = B_rowPtrs[k]; pb < B_rowPtrs[k + 1]; ++pb) {
            unsigned int j = B_colIdxs[pb];
            float prod = vA * B_vals[pb];

            int mj = m[j];
            bool is_new = (mj == -1);
            unsigned int idx;

            if (is_new) {
                idx = atomicAdd(pos_out, 1u);
                m[j] = idx;
                C_rowIdxs[idx] = i;
                C_colIdxs[idx] = j;
                C_vals[idx] = prod;
            } else {
                atomicAdd(&C_vals[mj], prod);
            }
        }
    }
}

void spmspm_gpu1(
    CSRMatrix *csrMatrix1,
    CSRMatrix *csrMatrix2,
    COOMatrix *cooMatrix3,
    unsigned int numRows1,
    unsigned int numRows2,
    unsigned int numCols2,
    unsigned int numNonzeros1,
    unsigned int numNonzeros2)
{
    // Extract CSR(A)
    CSRMatrix hA;
    CUDA_ERROR_CHECK(cudaMemcpy(&hA, csrMatrix1, sizeof(hA), cudaMemcpyDeviceToHost));
    const unsigned int *d_A_rowPtrs = hA.rowPtrs;
    const unsigned int *d_A_colIdxs = hA.colIdxs;
    const float *d_A_vals = hA.values;

    // Extract CSR(B)
    CSRMatrix hB;
    CUDA_ERROR_CHECK(cudaMemcpy(&hB, csrMatrix2, sizeof(hB), cudaMemcpyDeviceToHost));
    const unsigned int *d_B_rowPtrs = hB.rowPtrs;
    const unsigned int *d_B_colIdxs = hB.colIdxs;
    const float *d_B_vals = hB.values;

    // Extract COO(C)
    COOMatrix hC;
    CUDA_ERROR_CHECK(cudaMemcpy(&hC, cooMatrix3, sizeof(hC), cudaMemcpyDeviceToHost));
    unsigned int *d_C_rowIdxs = hC.rowIdxs;
    unsigned int *d_C_colIdxs = hC.colIdxs;
    float *d_C_vals = hC.values;

    int *d_marker;
    unsigned int *d_pos;
    CUDA_ERROR_CHECK(cudaMalloc(&d_marker, numRows1 * numCols2 * sizeof(int)));
    CUDA_ERROR_CHECK(cudaMalloc(&d_pos, sizeof(unsigned int)));
    CUDA_ERROR_CHECK(cudaMemset(d_pos, 0, sizeof(unsigned int)));

    int minGridSize = 0;
    int blockSize = 0;
    CUDA_ERROR_CHECK(cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, &blockSize, spmspm_gpu1_kernel, 0, 0));
    int gridSize = (numRows1 + blockSize - 1) / blockSize;

    // Launch kernel
    spmspm_gpu1_kernel<<<gridSize, blockSize>>>(
        numRows1,
        d_A_rowPtrs, d_A_colIdxs, d_A_vals,
        d_B_rowPtrs, d_B_colIdxs, d_B_vals,
        numCols2,
        d_C_rowIdxs, d_C_colIdxs, d_C_vals,
        d_marker,
        d_pos);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    // Copy number of nonzeros written back to device struct
    unsigned int newCount;
    CUDA_ERROR_CHECK(cudaMemcpy(&newCount, d_pos, sizeof(newCount), cudaMemcpyDeviceToHost));
    CUDA_ERROR_CHECK(cudaMemcpy(
        (char *)cooMatrix3 + offsetof(COOMatrix, numNonzeros),
        &newCount,
        sizeof(newCount),
        cudaMemcpyHostToDevice));

    // Free temporary storage
    CUDA_ERROR_CHECK(cudaFree(d_marker));
    CUDA_ERROR_CHECK(cudaFree(d_pos));
}

