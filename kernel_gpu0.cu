
#include <stdio.h>

#include "kernel.h"
#include "matrix.h"
#include "timer.h"

#define THRESHOLD 0.000001
#define YMAX 32

#define BLOCK_DIM 1024

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__ void histogram_gpu(unsigned int* rowIdxs_input, unsigned int* rowPtrs_result, unsigned int numRows_input, unsigned int nnz_input){

    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    unsigned int size = numRows_input;

    // --------- Histogram ---------
    __shared__ unsigned int bins_s[10000];
    if(threadIdx.x < size){
        bins_s[threadIdx.x] = 0;
    }
    __syncthreads();

    while(i < nnz_input){
        unsigned char b = rowIdxs_input[i];        
        atomicAdd(&bins_s[b], 1);
        i += stride;
    }
    __syncthreads();

    if(threadIdx.x < size){
        atomicAdd(&rowPtrs_result[threadIdx.x], bins_s[threadIdx.x]);
    }
}


__global__ void createCSRfromCOO(CSRMatrix* result, COOMatrix* A) {
    
    // Call histogram
    histogram_gpu(A->rowIdxs, result->rowPtrs, A->numRows, A->nnz);
    //cudaDeviceSynchronize();
	__syncthreads();

    // Prefix Sum
    if(threadIdx.x == 0){
        unsigned int sum = 0;
        for(unsigned int row = 0; row < A->numRows; ++row) {
            unsigned int val = result->rowPtrs[row];
            result->rowPtrs[row] = sum;
            sum += val;
        }
        result->rowPtrs[A->numRows] = sum;
    }

    // Binning
    for(unsigned int index = 0; index < A->nnz; ++index) {
        unsigned int row = A->rowIdxs[index];
        unsigned int i = result->rowPtrs[row]++;
        result->colIdxs[i] = A->colIdxs[index];
        result->values[i] = A->values[index];
    }

    // Restore row pointers
    for(unsigned int row = A->numRows - 1; row > 0; --row) {
        result->rowPtrs[row] = result->rowPtrs[row - 1];
    }
    result->rowPtrs[0] = 0;

    result->numRows = A->numRows;
    result->numCols = A->numCols;
    result->nnz = A->nnz;
    result->capacity = A->nnz;

}

__global__ void spmspm(COOMatrix *result, CSRMatrix *A, CSCMatrix *B, float bias, int offset) {

    unsigned int r = blockDim.x*blockIdx.x + threadIdx.x;
    unsigned int nnzIdx = 0;
    unsigned int temp;

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

                        __syncthreads();
                        
                        //Remove negative and zero values
                        if(sum > 0) {
                            if(sum>YMAX) {
                                sum = YMAX;
                            }
                            nnzIdx++;
                            temp = atomicAdd(&offset,1);
                            result->colIdxs[temp] = c;
                            result->values[temp] = sum;
                            result->rowIdxs[temp] =r ;
                        }    
                    }
                }
            }
        }
        atomicAdd(&result->nnz, nnzIdx);     
    }
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
    COOMatrix *outBufferCOO_d;
    CSRMatrix *outBufferCSR_d;
    CSCMatrix *W_d;

    // Allocate structs
    gpuErrchk( cudaMalloc((void **) &inBuffer_d, sizeof(CSRMatrix)));
    gpuErrchk( cudaMalloc((void **) &outBufferCSR_d, sizeof(CSRMatrix)));
    gpuErrchk( cudaMalloc((void **) &W_d, sizeof(CSCMatrix)));
    gpuErrchk( cudaMalloc((void **) &outBufferCOO_d, sizeof(COOMatrix)));
    
    // Allocate arrays
    // in buffer
   gpuErrchk( cudaMalloc((void **) &inBuffer_d->rowPtrs, inBuffer->numRows* sizeof(unsigned int)));
   gpuErrchk( cudaMalloc((void **) &inBuffer_d->colIdxs, inBuffer->numCols* sizeof(unsigned int)));
   gpuErrchk( cudaMalloc((void **) &inBuffer_d->values, inBuffer->capacity* sizeof(float)));
    
    // out bufferCSR
   gpuErrchk(  cudaMalloc((void **) &outBufferCSR_d->rowPtrs, outBufferCSR_d->numRows* sizeof(unsigned int)));
   gpuErrchk(  cudaMalloc((void **) &outBufferCSR_d->colIdxs, outBufferCSR_d->numCols* sizeof(unsigned int)));
   gpuErrchk(  cudaMalloc((void **) &outBufferCSR_d->values, outBufferCSR_d->capacity* sizeof(float)));

    // W
   gpuErrchk(  cudaMalloc((void **) &W_d->colPtrs, W_d->numCols* sizeof(unsigned int)));
   gpuErrchk(  cudaMalloc((void **) &W_d->rowIdxs, W_d->numRows* sizeof(unsigned int)));
   gpuErrchk(  cudaMalloc((void **) &W_d->values, W_d->capacity* sizeof(float)));

    // out bufferCOO
   gpuErrchk(  cudaMalloc((void **) &outBufferCOO_d->rowIdxs, outBufferCOO_d->numRows* sizeof(unsigned int)));
   gpuErrchk(  cudaMalloc((void **) &outBufferCOO_d->colIdxs, outBufferCOO_d->numCols* sizeof(unsigned int)));
   gpuErrchk(  cudaMalloc((void **) &outBufferCOO_d->values, outBufferCOO_d->capacity* sizeof(float)));

    // Copy data to gpu
    printf("Started copying");
    // Copy structs
    gpuErrchk(  cudaMemcpy(inBuffer_d, inBuffer, sizeof(CSRMatrix), cudaMemcpyHostToDevice));
    
    // Copy arrays
    // in buffer
   gpuErrchk(  cudaMemcpy(inBuffer_d->rowPtrs, inBuffer->rowPtrs, inBuffer->numRows* sizeof(unsigned int), cudaMemcpyHostToDevice));
   gpuErrchk(  cudaMemcpy(inBuffer_d->colIdxs, inBuffer->colIdxs, inBuffer->numCols*sizeof(unsigned int), cudaMemcpyHostToDevice));
   gpuErrchk(  cudaMemcpy(inBuffer_d->values, inBuffer->values, inBuffer->capacity*sizeof(float), cudaMemcpyHostToDevice));

    // Configurations
    const unsigned int threadsPerBlock = BLOCK_DIM;
    const unsigned int blocksPerGrid = (threadsPerBlock + outBuffer->numRows - 1)/threadsPerBlock;
        
    printf("before forloop");
    // Loop over layers
    for(unsigned int layer = 0; layer < numLayers; ++layer) {

        int offset = 0;        
        // Copy W data to gpu
       gpuErrchk( cudaMemcpy(W_d, W[layer], sizeof(CSRMatrix), cudaMemcpyHostToDevice));
       gpuErrchk( cudaMemcpy(W_d->colPtrs, W[layer]->colPtrs, W_d->numCols* sizeof(unsigned int), cudaMemcpyHostToDevice));
       gpuErrchk( cudaMemcpy(W_d->rowIdxs, W[layer]->rowIdxs, W_d->numRows* sizeof(unsigned int), cudaMemcpyHostToDevice));
       gpuErrchk( cudaMemcpy(W_d->colPtrs, W[layer]->values, W_d->capacity* sizeof(float), cudaMemcpyHostToDevice));
        
        // SpMSpM
        printf("Computing layer %u (SpMSpM)", layer);
        startTime(&timer);
        // spmspm(outBuffer, inBuffer, W[layer], bias);
        spmspm <<< blocksPerGrid, threadsPerBlock >>>(outBufferCOO_d, inBuffer_d, W_d, bias,offset);
        cudaDeviceSynchronize();

        stopTimeAndPrint(&timer, "");

        // printf("Computing layer %u (SpMSpM)", layer);
        startTime(&timer);
        createCSRfromCOO <<< blocksPerGrid, threadsPerBlock >>>(outBufferCSR_d, outBufferCOO_d);
        cudaDeviceSynchronize();
        stopTimeAndPrint(&timer, "");

        // Swap buffers
        CSRMatrix *t = inBuffer_d;
        inBuffer_d = outBufferCSR_d;
        outBufferCSR_d = t;
    }

    // Free data on GPU
    cudaMemcpy(inBuffer, inBuffer_d, sizeof(CSRMatrix), cudaMemcpyDeviceToHost);
    cudaMemcpy(inBuffer->rowPtrs, inBuffer_d->rowPtrs, inBuffer->numRows* sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(inBuffer->colIdxs,inBuffer_d->colIdxs,  inBuffer->numCols*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(inBuffer->values,inBuffer_d->values,  inBuffer->capacity*sizeof(float), cudaMemcpyDeviceToHost);

   
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
    cudaFree(inBuffer_d);
    cudaFree(outBufferCSR_d);
    cudaFree(outBufferCOO_d);
    cudaFree(W_d);

    stopTimeAndPrint(&timer, "Deallocate memory");

}
