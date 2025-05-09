#include "common.h"
#include <stddef.h>   
#include <stdio.h>    

// ─────────────────────────────────────────────────────────────────────────────
// Phase 1: Symbolic—count unique outputs per row using bit-packed shared marker
// ─────────────────────────────────────────────────────────────────────────────
__global__ __launch_bounds__(256) static void spmspm_gpu3_packed_symbolic(
    unsigned int rowsA,
    const unsigned int * __restrict__ A_rowPtrs,
    const unsigned int * __restrict__ A_colIdxs,
    const unsigned int * __restrict__ B_rowPtrs,
    const unsigned int * __restrict__ B_colIdxs,
    unsigned int colsB,
    unsigned int * __restrict__ rowCounts,
    unsigned int sharedWords        // = ceil(colsB/32)
) {
    extern __shared__ unsigned int packed[];  
    // each thread's slice = sharedWords words
    unsigned int *m = packed + threadIdx.x * sharedWords;

    // 1) clear bit-packed marker
    for (unsigned w = 0; w < sharedWords; ++w)
        m[w] = 0u;
    __syncthreads();

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rowsA) return;

    // count unique (i,j)
    unsigned int count = 0;
    unsigned int start = A_rowPtrs[i], end = A_rowPtrs[i+1];
    for (unsigned pa = start; pa < end; ++pa) {
        unsigned int k = A_colIdxs[pa];
        for (unsigned pb = B_rowPtrs[k]; pb < B_rowPtrs[k+1]; ++pb) {
            unsigned int j = B_colIdxs[pb];
            unsigned int word = j >> 5;         // j/32
            unsigned int bit  = j & 31;         // j%32
            unsigned int mask = 1u << bit;
            if ((m[word] & mask) == 0u) {
                m[word] |= mask;
                ++count;
            }
        }
    }
    rowCounts[i] = count;
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 2: Numeric—allocate exact slots then compute & write values
// ─────────────────────────────────────────────────────────────────────────────
__global__ __launch_bounds__(256) static void spmspm_gpu3_packed_numeric(
    unsigned int rowsA,
    const unsigned int * __restrict__ A_rowPtrs,
    const unsigned int * __restrict__ A_colIdxs,
    const float        * __restrict__ A_vals,
    const unsigned int * __restrict__ B_rowPtrs,
    const unsigned int * __restrict__ B_colIdxs,
    const float        * __restrict__ B_vals,
    unsigned int colsB,
    unsigned int * __restrict__ C_rowIdxs,
    unsigned int * __restrict__ C_colIdxs,
    float        * __restrict__ C_vals,
    const unsigned int * __restrict__ rowPtrsC,
    unsigned int sharedWords           // = ceil(colsB/32)
) {
    extern __shared__ unsigned int packed[];
    unsigned int *m = packed + threadIdx.x * sharedWords;

    // 1) clear marker again for numeric phase
    for (unsigned w = 0; w < sharedWords; ++w)
        m[w] = 0u;
    __syncthreads();

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rowsA) return;

    // write pointer for this row
    unsigned int writePtr = rowPtrsC[i];
    unsigned int start = A_rowPtrs[i], end = A_rowPtrs[i+1];
    for (unsigned pa = start; pa < end; ++pa) {
        unsigned int k = A_colIdxs[pa];
        float vA = A_vals[pa];
        for (unsigned pb = B_rowPtrs[k]; pb < B_rowPtrs[k+1]; ++pb) {
            unsigned int j = B_colIdxs[pb];
            float prod = vA * B_vals[pb];
            unsigned int word = j >> 5;
            unsigned int bit  = j & 31;
            unsigned int mask = 1u << bit;

            if ((m[word] & mask) == 0u) {
                // first time seeing (i,j)
                m[word] |= mask;
                C_rowIdxs[writePtr] = i;
                C_colIdxs[writePtr] = j;
                C_vals[writePtr]    = prod;
                ++writePtr;
            } else {
                // find the previous slot: must scan backward from rowPtrsC[i]
                // since we know it's already written, we can locate it by:
                //   linear search—colsB is small in “thin B” scenario
                unsigned int base = rowPtrsC[i];
                for (unsigned idx = base; idx < writePtr; ++idx) {
                    if (C_colIdxs[idx] == j) {
                        C_vals[idx] += prod;
                        break;
                    }
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Host‐side wrapper
// ─────────────────────────────────────────────────────────────────────────────
void spmspm_gpu3(
    COOMatrix *cooMatrix1,     // unused
    CSRMatrix *csrMatrix1,     // device CSR(A)
    CSCMatrix *cscMatrix1,     // unused
    COOMatrix *cooMatrix2,     // unused
    CSRMatrix *csrMatrix2,     // device CSR(B)
    CSCMatrix *cscMatrix2,     // unused
    COOMatrix *cooMatrix3,     // device COO(C)
    unsigned int numRows1,
    unsigned int numRows2,     // unused
    unsigned int numCols2,
    unsigned int numNonzeros1, // unused
    unsigned int numNonzeros2  // unused
) {
    // 1) Extract A pointers
    CSRMatrix hA;
    CUDA_ERROR_CHECK(cudaMemcpy(&hA, csrMatrix1, sizeof(hA), cudaMemcpyDeviceToHost));
    const unsigned int *d_A_rowPtrs = hA.rowPtrs;
    const unsigned int *d_A_colIdxs = hA.colIdxs;
    const float        *d_A_vals    = hA.values;

    // 2) Extract B pointers
    CSRMatrix hB;
    CUDA_ERROR_CHECK(cudaMemcpy(&hB, csrMatrix2, sizeof(hB), cudaMemcpyDeviceToHost));
    const unsigned int *d_B_rowPtrs = hB.rowPtrs;
    const unsigned int *d_B_colIdxs = hB.colIdxs;
    const float        *d_B_vals    = hB.values;

    // 3) Extract C pointers
    COOMatrix hC;
    CUDA_ERROR_CHECK(cudaMemcpy(&hC, cooMatrix3, sizeof(hC), cudaMemcpyDeviceToHost));
    unsigned int *d_C_rowIdxs = hC.rowIdxs;
    unsigned int *d_C_colIdxs = hC.colIdxs;
    float        *d_C_vals    = hC.values;

    // 4) Allocate rowCounts and rowPtrsC on device
    unsigned int *d_rowCounts, *d_rowPtrsC;
    CUDA_ERROR_CHECK(cudaMalloc(&d_rowCounts, numRows1 * sizeof(unsigned int)));
    CUDA_ERROR_CHECK(cudaMalloc(&d_rowPtrsC, (numRows1 + 1) * sizeof(unsigned int)));

    // 5) Launch symbolic kernel
    const int blockSize = 256;
    const int gridSize  = (numRows1 + blockSize - 1) / blockSize;
    // how many 32-bit words to hold colsB bits
    unsigned int sharedWords = (numCols2 + 31) / 32;
    size_t sharedBytes = blockSize * sharedWords * sizeof(unsigned int);

    spmspm_gpu3_packed_symbolic<<<gridSize, blockSize, sharedBytes>>>(
        numRows1,
        d_A_rowPtrs, d_A_colIdxs,
        d_B_rowPtrs, d_B_colIdxs,
        numCols2,
        d_rowCounts,
        sharedWords
    );
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    // 6) Host scan rowCounts → rowPtrsC
    unsigned int *h_counts = (unsigned int*)malloc(numRows1 * sizeof(unsigned int));
    unsigned int *h_ptrs   = (unsigned int*)malloc((numRows1 + 1) * sizeof(unsigned int));
    CUDA_ERROR_CHECK(cudaMemcpy(h_counts, d_rowCounts, numRows1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    h_ptrs[0] = 0;
    for (unsigned i = 0; i < numRows1; ++i)
        h_ptrs[i+1] = h_ptrs[i] + h_counts[i];
    CUDA_ERROR_CHECK(cudaMemcpy(d_rowPtrsC, h_ptrs, (numRows1 + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice));
    free(h_counts);
    free(h_ptrs);

    // 7) Launch numeric kernel
    spmspm_gpu3_packed_numeric<<<gridSize, blockSize, sharedBytes>>>(
        numRows1,
        d_A_rowPtrs, d_A_colIdxs, d_A_vals,
        d_B_rowPtrs, d_B_colIdxs, d_B_vals,
        numCols2,
        d_C_rowIdxs, d_C_colIdxs, d_C_vals,
        d_rowPtrsC,
        sharedWords
    );
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    // 8) Write back the total nnz into cooMatrix3
    unsigned int totalNNZ;
    CUDA_ERROR_CHECK(cudaMemcpy(&totalNNZ, d_rowPtrsC + numRows1, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_ERROR_CHECK(cudaMemcpy(
        (char*)cooMatrix3 + offsetof(COOMatrix, numNonzeros),
        &totalNNZ, sizeof(unsigned int), cudaMemcpyHostToDevice));

    // 9) Cleanup
    CUDA_ERROR_CHECK(cudaFree(d_rowCounts));
    CUDA_ERROR_CHECK(cudaFree(d_rowPtrsC));
}
