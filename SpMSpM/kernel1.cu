#include "common.h"
#include "matrix.h"
#include <stddef.h>
#include <stdio.h>

//
// Basic one‐thread‐per‐row CUDA kernel for SpMSpM (no optimizations).
// Each thread processes one row of A, multiplies it by B, and writes into C.
// “marker” prevents duplicate (i,j) slots; “pos_out” atomically reserves output indices.
//
__global__ static void spmspm_gpu0_kernel(
    unsigned int rowsA,            // number of rows in A
    const unsigned int *A_rowPtrs, // CSR row pointers for A
    const unsigned int *A_colIdxs, // CSR column indices for A
    const float *A_vals,           // CSR values for A
    const unsigned int *B_rowPtrs, // CSR row pointers for B
    const unsigned int *B_colIdxs, // CSR column indices for B
    const float *B_vals,           // CSR values for B
    unsigned int colsB,            // number of columns in B
    unsigned int *C_rowIdxs,       // output COO row indices
    unsigned int *C_colIdxs,       // output COO column indices
    float *C_vals,                 // output COO values
    int *marker,                   // temp “seen” array size rowsA×colsB
    unsigned int *pos_out          // atomic counter for output position
)
{
    // compute row index this thread will handle
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rowsA)
        return; // out-of-bounds guard

    // each thread clears its own marker[ i * colsB … i*colsB+colsB-1 ]
    int *m = marker + i * colsB;
    for (unsigned int j = 0; j < colsB; ++j)
        m[j] = -1;

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
            if (m[j] == -1)
            {
                // first time seeing (i,j): grab a slot
                unsigned int idx = atomicAdd(pos_out, 1u);
                C_rowIdxs[idx] = i;
                C_colIdxs[idx] = j;
                C_vals[idx] = prod;
                m[j] = idx;
            }
            else
            {
                // already have slot: accumulate
                C_vals[m[j]] += prod;
            }
        }
    }
}

void spmspm_gpu0(
    COOMatrix *cooMatrix1,     // device COO(A), not used directly here
    CSRMatrix *csrMatrix1,     // device CSR(A)
    CSCMatrix *cscMatrix1,     // device CSC(A), unused
    COOMatrix *cooMatrix2,     // device COO(B), unused
    CSRMatrix *csrMatrix2,     // device CSR(B)
    CSCMatrix *cscMatrix2,     // device CSC(B), unused
    COOMatrix *cooMatrix3,     // device COO(C)
    unsigned int numRows1,     // rows in A
    unsigned int numRows2,     // rows in B (unused)
    unsigned int numCols2,     // cols in B
    unsigned int numNonzeros1, // nnz in A (unused)
    unsigned int numNonzeros2  // nnz in B (unused)
)
{
    // ── 1) Pull out the raw device pointers from the CSR(A) struct ──
    CSRMatrix hA;
    CUDA_ERROR_CHECK(cudaMemcpy(&hA, csrMatrix1, sizeof(hA), cudaMemcpyDeviceToHost));
    const unsigned int *d_A_rowPtrs = hA.rowPtrs;
    const unsigned int *d_A_colIdxs = hA.colIdxs;
    const float *d_A_vals = hA.values;

    // ── 2) Pull out pointers from CSR(B) ──
    CSRMatrix hB;
    CUDA_ERROR_CHECK(cudaMemcpy(&hB, csrMatrix2, sizeof(hB), cudaMemcpyDeviceToHost));
    const unsigned int *d_B_rowPtrs = hB.rowPtrs;
    const unsigned int *d_B_colIdxs = hB.colIdxs;
    const float *d_B_vals = hB.values;

    // ── 3) Pull out pointers from COO(C) ──
    COOMatrix hC;
    CUDA_ERROR_CHECK(cudaMemcpy(&hC, cooMatrix3, sizeof(hC), cudaMemcpyDeviceToHost));
    unsigned int *d_C_rowIdxs = hC.rowIdxs;
    unsigned int *d_C_colIdxs = hC.colIdxs;
    float *d_C_vals = hC.values;

    // ── 4) Allocate two small helper arrays on device ──
    //    a) marker[rowsA*colsB]  b) pos_out (single unsigned int)
    int *d_marker;
    unsigned int *d_pos;
    CUDA_ERROR_CHECK(cudaMalloc(&d_marker, numRows1 * numCols2 * sizeof(int)));
    CUDA_ERROR_CHECK(cudaMalloc(&d_pos, sizeof(unsigned int)));
    CUDA_ERROR_CHECK(cudaMemset(d_pos, 0, sizeof(unsigned int)));

    // ── 5) Launch kernel: one thread per row of A ──
    const int threads = 512; // 512 threads per block
    const int blocks = (int)((numRows1 + threads - 1) / threads);
    spmspm_gpu0_kernel<<<blocks, threads>>>(
        numRows1,
        d_A_rowPtrs, d_A_colIdxs, d_A_vals,
        d_B_rowPtrs, d_B_colIdxs, d_B_vals,
        numCols2,
        d_C_rowIdxs, d_C_colIdxs, d_C_vals,
        d_marker,
        d_pos);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    // ── 6) Copy the final count back into the device COO(C) struct ──
    unsigned int newCount;
    CUDA_ERROR_CHECK(cudaMemcpy(&newCount, d_pos, sizeof(newCount), cudaMemcpyDeviceToHost));
    CUDA_ERROR_CHECK(cudaMemcpy(
        (char *)cooMatrix3 + offsetof(COOMatrix, numNonzeros),
        &newCount,
        sizeof(newCount),
        cudaMemcpyHostToDevice));

    // ── 7) Free only our temporaries ──
    CUDA_ERROR_CHECK(cudaFree(d_marker));
    CUDA_ERROR_CHECK(cudaFree(d_pos));
}