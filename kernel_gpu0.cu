
#include <stdio.h>

#include "kernel.h"
#include "matrix.h"
#include "timer.h"

#define THRESHOLD 0.000001
#define YMAX 32

#define BLOCK_DIM 1024

__device__ void histogram(CSRMatrix* result, COOMatrix *A) {

    unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    unsigned int size = A->numRows;

    // --------- Histogram ---------
    __shared__ unsigned int bins_s[size];
    if(threadIdx.x < size){
        bins_s[threadIdx.x] = 0;
    }
    __syncthreads();

    while(i < A->nnz){
        unsigned char b = A->rowIdxs[i];        
        atomicAdd(&bins_s[b], 1);
        i += stride;
    }
    __syncthreads();

    if(threadIdx.x < size){
        atomicAdd(&result->rowPtrs[threadIdx.x], bins_s[threadIdx.x]);
    }
}

__device__ void prefixSum(CSRMatrix* result, COOMatrix* A) {

    unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;
    
    // --------- Kogge-Stone Exclusive -------


    /*
    // Prefix sum
    unsigned int sum = 0;
    for(unsigned int row = 0; row < A->numRows; ++row) {
        unsigned int val = rowPtrs[row];
        rowPtrs[row] = sum;
        sum += val;
    }
    rowPtrs[A->numRows] = sum;
    */
}

__device__ void createCSRfromCOO(CSRMatrix* result, COOMatrix* A) {
    histogram(result, A);

    prefixSum(result, A);

}

__global__ void spmspm(CSRMatrix *result, CSRMatrix *A, CSCMatrix *B, float bias) {

    unsigned int r = blockDim.x*blockIdx.x + threadIdx.x;
    unsigned int nnzIdx0 = 0;
    unsigned int nnzIdx1 = 0; 
    __shared__ int offset[BLOCK_DIM];

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

                        // __syncthreads();
                        
                        //Remove negative and zero values
                        if(sum > 0) {
                            if(sum>YMAX) {
                                sum = YMAX;
                            }
                            ++nnzIdx0;
                        }    
                    }
                }
            }
        }
        offset[r] = nnzIdx0;        
    }
    __syncthreads();
    
    // Prefix sum
    if(threadIdx.x == 0){
        if(r == 0){
            offset[r] = 0;
        }
        else{
            offset[0] = result->nnz;
        }
        for(int i = 1; i<BLOCK_DIM; ++i){
            offset[i] += offset[i-1];
        }
    }
    __syncthreads();
    

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
                            ++nnzIdx1;
                            result->colIdxs[nnzIdx1 + x] = c;
                            result->values[nnzIdx1 + x] = sum;
                        }    
                    }
                }
                result->rowPtrs[r + 1] = x + nnzIdx1; 
            }
        }
        // result->nnz = nnzIdx;  
        atomicAdd(&result->nnz, nnzIdx1);     
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
    CSRMatrix *outBuffer_d;
    CSCMatrix *W_d;

    cudaMalloc((void **) &inBuffer_d, sizeof(CSRMatrix));
    cudaMalloc((void **) &outBuffer_d, sizeof(CSRMatrix));
    cudaMalloc((void **) &W_d, sizeof(CSCMatrix));
    

    // Loop over layers
    for(unsigned int layer = 0; layer < numLayers; ++layer) {

        // Configurations
        const unsigned int threadsPerBlock = BLOCK_DIM;
        const unsigned int blocksPerGrid = (threadsPerBlock + outBuffer->numRows - 1)/threadsPerBlock;

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
