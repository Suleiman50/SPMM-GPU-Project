#include "common.h"
#include <stddef.h>
#include <stdio.h>

// ------------------------
// Two-phase approach with shared-memory markers (no global marker array)
// ------------------------

// Symbolic kernel: one thread per row, using shared marker array
__global__ __launch_bounds__(512) static void spmspm_gpu3_symbolic(
    unsigned int rowsA,
    const unsigned int *A_rowPtrs,
    const unsigned int *A_colIdxs,
    const float *A_vals,
    const unsigned int *B_rowPtrs,
    const unsigned int *B_colIdxs,
    const float *B_vals,
    unsigned int colsB,
    unsigned int *rowCounts)
{
    extern __shared__ int marker[]; // size = blockDim.x * colsB
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rowsA)
        return;

    // Per-thread shared marker slice
    int *m = marker + threadIdx.x * colsB;
    // Clear marker in shared memory
    for (unsigned j = 0; j < colsB; ++j)
        m[j] = -1;
    __syncthreads();

    unsigned int cnt = 0;
    unsigned int r0 = A_rowPtrs[i];
    unsigned int r1 = A_rowPtrs[i + 1];
    for (unsigned pa = r0; pa < r1; ++pa)
    {
        unsigned int k = A_colIdxs[pa];
        for (unsigned pb = B_rowPtrs[k]; pb < B_rowPtrs[k + 1]; ++pb)
        {
            unsigned int j = B_colIdxs[pb];
            if (m[j] == -1)
            {
                m[j] = 1;
                ++cnt;
            }
        }
    }
    rowCounts[i] = cnt;
}

// Numeric kernel: one thread per row, using shared marker array
__global__ __launch_bounds__(512) static void spmspm_gpu3_numeric(
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
    const unsigned int *rowPtrsC)
{
    extern __shared__ int marker[]; // size = blockDim.x * colsB
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rowsA)
        return;

    // Per-thread shared marker slice
    int *m = marker + threadIdx.x * colsB;
    // Clear marker
    for (unsigned j = 0; j < colsB; ++j)
        m[j] = -1;
    __syncthreads();

    unsigned int writePtr = rowPtrsC[i];
    unsigned int r0 = A_rowPtrs[i];
    unsigned int r1 = A_rowPtrs[i + 1];
    for (unsigned pa = r0; pa < r1; ++pa)
    {
        unsigned int k = A_colIdxs[pa];
        float vA = A_vals[pa];
        for (unsigned pb = B_rowPtrs[k]; pb < B_rowPtrs[k + 1]; ++pb)
        {
            unsigned int j = B_colIdxs[pb];
            float prod = vA * B_vals[pb];
            int &slot = m[j];
            if (slot == -1)
            {
                slot = writePtr;
                C_rowIdxs[writePtr] = i;
                C_colIdxs[writePtr] = j;
                C_vals[writePtr] = prod;
                ++writePtr;
            }
            else
            {
                C_vals[slot] += prod;
            }
        }
    }
}

void spmspm_gpu3(
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
    // 1) Extract device pointers
    CSRMatrix hA;
    CUDA_ERROR_CHECK(cudaMemcpy(&hA, csrMatrix1, sizeof(hA), cudaMemcpyDeviceToHost));
    CSRMatrix hB;
    CUDA_ERROR_CHECK(cudaMemcpy(&hB, csrMatrix2, sizeof(hB), cudaMemcpyDeviceToHost));
    COOMatrix hC;
    CUDA_ERROR_CHECK(cudaMemcpy(&hC, cooMatrix3, sizeof(hC), cudaMemcpyDeviceToHost));
    const unsigned int *d_A_rowPtrs = hA.rowPtrs;
    const unsigned int *d_A_colIdxs = hA.colIdxs;
    const float *d_A_vals = hA.values;
    const unsigned int *d_B_rowPtrs = hB.rowPtrs;
    const unsigned int *d_B_colIdxs = hB.colIdxs;
    const float *d_B_vals = hB.values;
    unsigned int *d_C_rowIdxs = hC.rowIdxs;
    unsigned int *d_C_colIdxs = hC.colIdxs;
    float *d_C_vals = hC.values;

    // 2) Allocate rowCounts and rowPtrsC
    unsigned int *d_rowCounts, *d_rowPtrsC;
    CUDA_ERROR_CHECK(cudaMalloc(&d_rowCounts, numRows1 * sizeof(unsigned int)));
    CUDA_ERROR_CHECK(cudaMalloc(&d_rowPtrsC, (numRows1 + 1) * sizeof(unsigned int)));

    // 3) Launch symbolic kernel with shared memory for marker
    int blockSize = 256;
    int gridSize = (numRows1 + blockSize - 1) / blockSize;
    size_t sharedBytes = blockSize * numCols2 * sizeof(int);
    spmspm_gpu3_symbolic<<<gridSize, blockSize, sharedBytes>>>(
        numRows1,
        d_A_rowPtrs, d_A_colIdxs, d_A_vals,
        d_B_rowPtrs, d_B_colIdxs, d_B_vals,
        numCols2,
        d_rowCounts);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    // 4) Host scan of rowCounts -> rowPtrsC
    unsigned int *h_counts = (unsigned int *)malloc(numRows1 * sizeof(unsigned int));
    unsigned int *h_ptrs = (unsigned int *)malloc((numRows1 + 1) * sizeof(unsigned int));
    CUDA_ERROR_CHECK(cudaMemcpy(h_counts, d_rowCounts, numRows1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    h_ptrs[0] = 0;
    for (unsigned i = 0; i < numRows1; ++i)
        h_ptrs[i + 1] = h_ptrs[i] + h_counts[i];
    CUDA_ERROR_CHECK(cudaMemcpy(d_rowPtrsC, h_ptrs, (numRows1 + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice));
    free(h_counts);
    free(h_ptrs);

    // 5) Launch numeric kernel with shared-memory for marker
    spmspm_gpu3_numeric<<<gridSize, blockSize, sharedBytes>>>(
        numRows1,
        d_A_rowPtrs, d_A_colIdxs, d_A_vals,
        d_B_rowPtrs, d_B_colIdxs, d_B_vals,
        numCols2,
        d_C_rowIdxs, d_C_colIdxs, d_C_vals,
        d_rowPtrsC);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    // 6) Write back nnz count
    unsigned int totalNNZ;
    CUDA_ERROR_CHECK(cudaMemcpy(&totalNNZ, d_rowPtrsC + numRows1, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_ERROR_CHECK(cudaMemcpy((char *)cooMatrix3 + offsetof(COOMatrix, numNonzeros),
                                &totalNNZ, sizeof(unsigned int), cudaMemcpyHostToDevice));

    // Cleanup
    CUDA_ERROR_CHECK(cudaFree(d_rowCounts));
    CUDA_ERROR_CHECK(cudaFree(d_rowPtrsC));
}