
#include <stdio.h>

#include "kernel.h"
#include "matrix.h"
#include "timer.h"

#define THRESHOLD 0.000001
#define YMAX 32

#define BLOCK_DIM 1024

__global__ spmspm(CSRMatrix *result, CSRMatrix *A, CSCMatrix *B, float bias) {

    unsigned int r = blockDim.x*blockIdx.x + threadIdx.x;
    unsigned int nnzIdx; // Number of non zero indexes in Result
    if(r > 1){
        nnzIdx = result->rowPtrs[r - 1];
    } else {
        nnzIdx = 0;
    }

    if(r < A->numRows ){
        unsigned int rowPtrA = A->rowPtrs[r]; // Index of the current rowPtrs element
        unsigned int nnzA = A->rowPtrs[r + 1] - rowPtrA;  // Number of non zero elements in A

        if(nnzA > 0){
            unsigned int *colIdxsA = A->colIdxs + rowPtrA;
            float *valueA = A->values + rowPtrA;

            // Loop over B columns
            for(unsigned int c = 0; c < B->numCols; ++c){
                unsigned int colPtrB = B->colPtrs[c];
                unsigned int nnzB = B->colPtrs[c + 1] = colPtrB;

                if(nnzB > 0){
                    unsigned int *rowsIdxsB = B->rowsIdxs + colPtrB;
                    float *valueB = B->values + colPtrB;

                    // Loop and find intersection
                    float sum = 0;
                    unsigned int ia = 0;
                    unsigned int ib = 0;

                    // Loop over segment of non zero elements in the row of A and col of B
                    while(ia < nnzA && ib < nnzzB){
                        unsigned int colIdx = colIdxsA[ia];
                        unsigned int rowIdx = rowsIdxsB[ib];
                        if(colIdx < rowIdx) {
                            ia++;
                        } else if(colIdx > rowIdx) {
                            ib++;
                        } else {
                            sum += valueA[ia]*valueB[ib];
                            ia++;
                            ib++;
                        }
                    }

                    // Sync threads
                    __syncthreads();

                    // Write to Result
                    if(sum > THRESHOLD || sum < -THRESHOLD) {
                        sum += bias;

                        __syncthreads();
                        
                        //Remove negative and zero values
                        if(sum > 0) {
                            if(sum>YMAX) {
                                sum = YMAX;
                            }
                            if(nnzIdx >= result->capacity) {
                                expandCSRCapacity(result, 2*result->capacity);
                            }
                            result->colIdxs[nnzIdx] = c; // FIX HERE
                            result->values[nnzIdx] = sum;
                            ++nnzIdx;
                        }    
                    }
                }
            }
        }
        result->rowPtrs[r + 1] = nnzIdx;        
    }
    // Atomic add (?)
    atomicAdd(&result->nnz, nnzIdx);
    // result->nnz = nnzIdx;

}

void findNonzeroRows(Vector* v, CSRMatrix* A) {
    unsigned int nnz = 0;
    for(unsigned int r = 0; r < A->numRows; ++r) {
        unsigned int rowPtrA = A->rowPtrs[r];
        unsigned int nnzA = A->rowPtrs[r + 1] - rowPtrA;
        if(nnzA > 0) {
            if(nnz >= v->capacity) {
                expandVectorCapacity(v, 2*v->capacity);
            }
            v->data[nnz] = r;
            ++nnz;
        }
    }
    v->nnz = nnz;
}

void sparseNN(Vector* result, COOMatrix* featureVectors, COOMatrix** layerWeights, float bias, unsigned int numLayers) {


    Timer timer;

    // Convert featureVectors to CSR
    startTime(&timer);
    CSRMatrix* Y0 = createCSRfromCOO(featureVectors);
    stopTimeAndPrint(&timer, "Convert feature vectors to CSR");

    // Convert layer weights to CSC
    startTime(&timer);
    CSCMatrix* W[numLayers];
    for(unsigned int layer = 0; layer < numLayers; ++layer) {
        W[layer] = createCSCfromCOO(layerWeights[layer]);
    }
    stopTimeAndPrint(&timer, "Convert weights to CSR");

    // Double buffers
    startTime(&timer);
    CSRMatrix *tmp = createEmptyCSR(Y0->numRows, Y0->numCols, 2*Y0->nnz);
    CSRMatrix *inBuffer  = Y0;
    CSRMatrix *outBuffer = tmp;
    stopTimeAndPrint(&timer, "Allocate temporary buffer");
        
    // Allocate memory on GPU
    CSRMatrix *inBuffer_d;
    CSRMatrix *outBuffer_d;
    CSCMatrix *W_d;

    cudaMalloc((void **) &inBuffer_d, sizeof(CSRMatrix));
    cudaMalloc((void **) &outBuffer_d, sizeof(CSRMatrix));
    cudaMalloc((void **) &W_d, sizeof(CSCMatrix));
    

    // Loop over layers
    for(unsigned int layer = 0; layer < numLayers; ++layer) {

        // Configurations
        const unsigned int threadsPerBlock = BLOCK_DIM;
        const unsigned int blocksPerGrid = (threadsPerBlock + outBuffer->numRows() - 1)/threadsPerBlock;

        // Copy data to gpu
        cudaMemcpy(inBuffer_d, inBuffer, sizeof(CSRMatrix), cudaMemcpyHostToDevice);
        cudaMemcpy(outBuffer_d, outBuffer, sizeof(CSRMatrix), cudaMemcpyHostToDevice);
        cudaMemcpy(W_d, W[layer], sizeof(CSRMatrix), cudaMemcpyHostToDevice);

        // SpMSpM
        printf("Computing layer %u (SpMSpM)", layer);
        startTime(&timer);
        // spmspm(outBuffer, inBuffer, W[layer], bias);
        spmspm <<< blocksPerGrid, threadsPerBlock >>>(outBuffer_d, inBuffer_d, W_d, bias);
        stopTimeAndPrint(&timer, "");

        // Swap buffers
        CSRMatrix *t = inBuffer;
        inBuffer = outBuffer;
        outBuffer = t;

        
    }
    // Free data on GPU
    cudaFree(inBuffer_d);
    cudaFree(outBuffer_d);
    cudaFree(W_d);

    // Find nonzero rows
    startTime(&timer);
    findNonzeroRows(result, inBuffer);
    stopTimeAndPrint(&timer, "Find nonzero rows");

    // Free buffers
    startTime(&timer);
    freeCSR(Y0);
    for(unsigned int layer = 0; layer < numLayers; ++layer) {
        freeCSC(W[layer]);
    }
    freeCSR(tmp);
    stopTimeAndPrint(&timer, "Deallocate memory");

}

