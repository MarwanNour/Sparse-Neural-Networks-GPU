
#include <stdio.h>

#include "kernel.h"
#include "matrix.h"
#include "timer.h"

#define THRESHOLD 0.000001
#define YMAX 32

#define BLOCK_DIM 1024

__global__ void spmspm(CSRMatrix *result, CSRMatrix *A, CSCMatrix *B, float bias, int offset) {

    unsigned int r = blockDim.x*blockIdx.x + threadIdx.x;
    unsigned int nnzIdx = 0;
    unsigned int temp;

    if(r < A->numRows ){
        unsigned int x=offset[r];
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
                    unsigned int *rowIdxsB = B->rowIdxs + colPtrB;
                    float *valueB = B->values + colPtrB;

                    // Loop and find intersection
                    float sum = 0;
                    unsigned int ia = 0;
                    unsigned int ib = 0;

                    // Loop over segment of non zero elements in the row of A and col of B
                    while(ia < nnzA && ib < nnzB){
                        unsigned int colIdx = colIdxsA[ia];
                        unsigned int rowIdx = rowIdxsB[ib];
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
                    // Write to Result
                    if(sum > THRESHOLD || sum < -THRESHOLD) {
                        sum += bias;

                        __syncthreads();
                        
                        //Remove negative and zero values
                        if(sum > 0) {
                            if(sum>YMAX) {
                                sum = YMAX;
                            }
                            nnzIdx++;
                            temp= atomicAdd(offset,1);
                            result->colIdxs[temp] = c;
                            result->values[temp] = sum;
                            result->rowIdxs[temp] =r ;
                        }    
                    }
                }
                // result->rowPtrs[r + 1] = x + temp; 
            }
        }
        // result->nnz = nnzIdx;  
        atomicAdd(&result->nnz, nnzIdx);     
    }

    // __syncthreads();


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
    CSRMatrix *outBufferCOO_d;
    COOMatrix *outBuffer_d;
    CSCMatrix *W_d;

    cudaMalloc((void **) &inBuffer_d, sizeof(CSRMatrix));
    cudaMalloc((void **) &outBufferCSR_d, sizeof(CSRMatrix));
    cudaMalloc((void **) &W_d, sizeof(CSCMatrix));
    cudaMalloc((void **) &outBufferCOO_d, sizeof(COOMatrix));

    cudaMemcpy(inBuffer_d, inBuffer, sizeof(CSRMatrix), cudaMemcpyHostToDevice);
    // cudaMemcpy(outBuffer_d, outBuffer, sizeof(CSRMatrix), cudaMemcpyHostToDevice);
    // Loop over layers
    for(unsigned int layer = 0; layer < numLayers; ++layer) {

        // Configurations
        const unsigned int threadsPerBlock = BLOCK_DIM;
        const unsigned int blocksPerGrid = (threadsPerBlock + outBuffer->numRows - 1)/threadsPerBlock;
        int offset=0;
        // Copy data to gpu
        cudaMemcpy(W_d, W[layer], sizeof(CSRMatrix), cudaMemcpyHostToDevice);

        // SpMSpM
        printf("Computing layer %u (SpMSpM)", layer);
        startTime(&timer);
        // spmspm(outBuffer, inBuffer, W[layer], bias);
        spmspm <<< blocksPerGrid, threadsPerBlock >>>(outBufferCOO_d, inBuffer_d, W_d, bias,offset);
        stopTimeAndPrint(&timer, "");

        // printf("Computing layer %u (SpMSpM)", layer);
        startTime(&timer);
        CooToCSR <<< blocksPerGrid, threadsPerBlock >>>(outBuffer_d, outBuffer1_d);
        // createCSRfromCOO(outBufferCSR_d,outBufferCOO_d)
        stopTimeAndPrint(&timer, "");

        // Swap buffers
        CSRMatrix *t = inBuffer_d;
        inBuffer_d = outBufferCSR_d;
        outBufferCSR_d = t;

        
    }
    // Free data on GPU
    cudaMemcpy(inBuffer, inBuffer_d, sizeof(CSRMatrix), cudaMemcpyDeviceToHost);
    // cudaMemcpy(outBuffer, inBuffer_d, sizeof(CSRMatrix), cudaMemcpyDeviceToHost);



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
