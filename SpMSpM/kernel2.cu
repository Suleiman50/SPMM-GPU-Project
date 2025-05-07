// Optimization: Remove per-thread marker initialization and use a single cudaMemset to clear the marker array

#include "common.h"
#include <stddef.h> // for offsetof
#include <stdio.h>

// Basic one‐thread‐per‐row CUDA kernel for SpMSpM, without per-thread clearing
// marker array is assumed pre-cleared (all -1) via cudaMemset before launch.
__global__ __launch_bounds__(512) static void spmspm_gpu2_kernel(
    unsigned int rowsA,
    const unsigned int *__restrict__ A_rowPtrs,
    const unsigned int *__restrict__ A_colIdxs,
    const float *__restrict__ A_vals,
    const unsigned int *__restrict__ B_rowPtrs,
    const unsigned int *__restrict__ B_colIdxs,
    const float *__restrict__ B_vals,
    unsigned int colsB,
    unsigned int *__restrict__ C_rowIdxs,
    unsigned int *__restrict__ C_colIdxs,
    float *__restrict__ C_vals,
    int *__restrict__ marker,
    unsigned int *pos_out)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rowsA)
        return;

    int *m = marker + i * colsB;
    // marker already memset to -1, so we skip clearing here

    // for each nonzero A(i,k)
    for (unsigned int pa = A_rowPtrs[i]; pa < A_rowPtrs[i + 1]; ++pa)
    {
        unsigned int k = A_colIdxs[pa];
        float vA = A_vals[pa];
        // for each nonzero B(k,j)
        for (unsigned int pb = B_rowPtrs[k]; pb < B_rowPtrs[k + 1]; ++pb)
        {
            unsigned int j = B_colIdxs[pb];
            float prod = vA * B_vals[pb];
            int mj = m[j];
            if (mj == -1)
            {
                unsigned int idx = atomicAdd(pos_out, 1u);
                m[j] = idx;
                C_rowIdxs[idx] = i;
                C_colIdxs[idx] = j;
                C_vals[idx] = prod;
            }
            else
            {
                atomicAdd(&C_vals[mj], prod);
            }
        }
    }
}

void spmspm_gpu2(
    COOMatrix *cooMatrix1,
    CSRMatrix *csrMatrix1,
    CSCMatrix *cscMatrix1,
    COOMatrix *cooMatrix2,
    CSRMatrix *csrMatrix2,
    CSCMatrix *cscMatrix2,
    COOMatrix *cooMatrix3,
    unsigned int numRows1,
    unsigned int numRows2,
    unsigned int numCols2,
    unsigned int numNonzeros1,
    unsigned int numNonzeros2)
{
    // Extract pointers from CSR(A)
    CSRMatrix hA;
    CUDA_ERROR_CHECK(cudaMemcpy(&hA, csrMatrix1, sizeof(hA), cudaMemcpyDeviceToHost));
    const unsigned int *d_A_rowPtrs = hA.rowPtrs;
    const unsigned int *d_A_colIdxs = hA.colIdxs;
    const float *d_A_vals = hA.values;

    // Extract pointers from CSR(B)
    CSRMatrix hB;
    CUDA_ERROR_CHECK(cudaMemcpy(&hB, csrMatrix2, sizeof(hB), cudaMemcpyDeviceToHost));
    const unsigned int *d_B_rowPtrs = hB.rowPtrs;
    const unsigned int *d_B_colIdxs = hB.colIdxs;
    const float *d_B_vals = hB.values;

    // Extract pointers from COO(C)
    COOMatrix hC;
    CUDA_ERROR_CHECK(cudaMemcpy(&hC, cooMatrix3, sizeof(hC), cudaMemcpyDeviceToHost));
    unsigned int *d_C_rowIdxs = hC.rowIdxs;
    unsigned int *d_C_colIdxs = hC.colIdxs;
    float *d_C_vals = hC.values;

    // Allocate and clear marker and pos counter
    int *d_marker;
    unsigned int *d_pos;
    CUDA_ERROR_CHECK(cudaMalloc(&d_marker, numRows1 * numCols2 * sizeof(int)));
    CUDA_ERROR_CHECK(cudaMalloc(&d_pos, sizeof(unsigned int)));
    // Bulk-clear marker[] to -1 in one go (optimization)
    CUDA_ERROR_CHECK(cudaMemset(d_marker, 0xFF, numRows1 * numCols2 * sizeof(int)));
    CUDA_ERROR_CHECK(cudaMemset(d_pos, 0, sizeof(unsigned int)));

    // Launch parameters
    const int threads = 512;
    const int blocks = (numRows1 + threads - 1) / threads;

    // Launch optimized kernel
    spmspm_gpu2_kernel<<<blocks, threads>>>(
        numRows1,
        d_A_rowPtrs, d_A_colIdxs, d_A_vals,
        d_B_rowPtrs, d_B_colIdxs, d_B_vals,
        numCols2,
        d_C_rowIdxs, d_C_colIdxs, d_C_vals,
        d_marker,
        d_pos);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    // Copy number of nonzeros back to cooMatrix3
    unsigned int newCount;
    CUDA_ERROR_CHECK(cudaMemcpy(&newCount, d_pos, sizeof(newCount), cudaMemcpyDeviceToHost));
    CUDA_ERROR_CHECK(cudaMemcpy(
        (char *)cooMatrix3 + offsetof(COOMatrix, numNonzeros),
        &newCount,
        sizeof(newCount),
        cudaMemcpyHostToDevice));

    // Free temporaries
    CUDA_ERROR_CHECK(cudaFree(d_marker));
    CUDA_ERROR_CHECK(cudaFree(d_pos));
}