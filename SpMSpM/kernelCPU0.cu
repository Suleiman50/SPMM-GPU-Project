#include "common.h"
#include <stdio.h>
#include <stdlib.h>
void spmspm_cpu0(
    COOMatrix *cooMatrix1,
    CSRMatrix *csrMatrix1,
    CSCMatrix *cscMatrix1,
    COOMatrix *cooMatrix2,
    CSRMatrix *csrMatrix2,
    CSCMatrix *cscMatrix2,
    COOMatrix *cooMatrix3)
{
    // clear output COO
    clearCOOMatrix(cooMatrix3);

    unsigned int rowsA = csrMatrix1->numRows;
    unsigned int colsB = csrMatrix2->numCols;
    unsigned int pos = 0;

    // marker[j] < rowStart means (i,j) hassn't been emitted yet
    int *marker = (int *)malloc(colsB * sizeof(int));
    for (unsigned int j = 0; j < colsB; ++j)
    {
        marker[j] = -1;
    }

    for (unsigned int i = 0; i < rowsA; ++i)
    {
        unsigned int rowStart = pos;

        // for each nonzero A(i,k)
        for (unsigned int pa = csrMatrix1->rowPtrs[i];
             pa < csrMatrix1->rowPtrs[i + 1];
             ++pa)
        {
            unsigned int k = csrMatrix1->colIdxs[pa];
            float vA = csrMatrix1->values[pa];

            // for each nonzero B(k,j)
            for (unsigned int pb = csrMatrix2->rowPtrs[k];
                 pb < csrMatrix2->rowPtrs[k + 1];
                 ++pb)
            {
                unsigned int j = csrMatrix2->colIdxs[pb];
                float vB = csrMatrix2->values[pb];
                float prod = vA * vB;

                if (marker[j] < (int)rowStart)
                {
                    // first contribution to C(i,j)
                    marker[j] = pos;
                    cooMatrix3->rowIdxs[pos] = i;
                    cooMatrix3->colIdxs[pos] = j;
                    cooMatrix3->values[pos] = prod;
                    ++pos;
                }
                else
                {
                    // accumulate into existing slot
                    cooMatrix3->values[marker[j]] += prod;
                }
            }
        }
    }

    cooMatrix3->numNonzeros = pos;
    free(marker);
}
