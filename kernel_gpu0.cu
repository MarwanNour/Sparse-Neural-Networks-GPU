
#include <stdio.h>

#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#include "kernel.h"
#include "matrix.h"
#include "timer.h"

#define THRESHOLD 0.000001
#define YMAX 32

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
} 
#define PRIVATE 4096
#define BLOCK_DIM 1024
// unsigned int* input, unsigned int* bins, unsigned int numElems
__global__ void histogram_gpu(CSRMatrix* result, COOMatrix* A){
    // {


        unsigned int* input ;
        unsigned int* bins;
        unsigned int numElems;
        input=A->rowIdxs;
        bins=result->rowPtrs;
        numElems=A->nnz;
        int tx = threadIdx.x; int bx = blockIdx.x;
        
        // compute global thread coordinates
        int i = (bx * blockDim.x) + tx;
        // if (i==0){printf(" histo nnz=%d",A->nnz);}

    // create a private histogram copy for each thread block
    __shared__ unsigned int hist[PRIVATE];
    if(bx==0){
        for (int j=tx; j<A->numRows; j+=BLOCK_DIM) {
                bins[j] = 0;
        }
    }

   

    // each thread must initialize more than 1 location
    if (PRIVATE > BLOCK_DIM) {
        for (int j=tx; j<PRIVATE; j+=BLOCK_DIM) {
            if (j < PRIVATE) {
                hist[j] = 0;
            }
        }
    }
    // use the first `PRIVATE` threads of each block to init
    else {
        if (tx < PRIVATE) {
            hist[tx] = 0;
        }
    }
    // wait for all threads in the block to finish
    __syncthreads();

    // update private histogram
    if (i < numElems) {
        if(input[i]<PRIVATE){
        atomicAdd(&(hist[input[i]]), 1);}
        else{
            atomicAdd(&(bins[input[i]]), 1);
        }
    }
    // wait for all threads in the block to finish
    __syncthreads();

    // each thread must update more than 1 location
    if (PRIVATE > BLOCK_DIM) {
        for (int j=tx; j<PRIVATE; j+=BLOCK_DIM) {
            if (j < PRIVATE) {
                atomicAdd(&(bins[j]), hist[j]);
            }
        }
    }
    // use the first `PRIVATE` threads to update final histogram
    else {
        if (tx < PRIVATE) {
            atomicAdd(&(bins[tx]), hist[tx]);
            
        }
    }
    // for(unsigned int i = 0; i < A->numRows+1; ++i) {
    //     // unsigned int row = A->rowIdxs[i];
    //     result->rowPtrs[i]=0;
    // }
    
    // for(unsigned int i = 0; i < A->nnz; ++i) {
    //             unsigned int row = A->rowIdxs[i];
    //             result->rowPtrs[row]++;
    //         }
}

__device__ void sort_by_key(unsigned int * keys,float * values,int size){
    int i, j,temp;
    float temp2;  
    for (i = 0; i < size-1; i++)   {   
      
    // Last i elements are already in place  
    for (j = 0; j < size-i-1; j++)  {
        if (keys[j] > keys[j+1]){
             temp = keys[j+ 1];  
            keys[j+1] = keys[j];  
            keys[j]= temp; 
            temp2 = keys[j+ 1];  
            keys[j+1] = keys[j];  
            keys[j]= temp2;  

            }
        }
    }
}
__global__ void Binning(CSRMatrix *result ,COOMatrix *A ){
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
    result->capacity = A->capacity;
}

__global__ void createCSRfromCOO_gpu(CSRMatrix* result, COOMatrix* A) {


    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    // if(i==0){
    //     printf("nnz=%d\n",A->nnz);
    //     for(unsigned int i = 0; i < A->nnz; ++i) {
    //         unsigned int row = A->rowIdxs[i];
    //         result->rowPtrs[row]++;
    //     }


    // }
    // if(i==0){
    //     result->rowPtrs=0;
    // }
    // if(i<A->numRows){
    //     int count = thrust::count(thrust::device ,A->rowIdxs, A->rowIdxs+A->nnz, i);
    //     result->rowPtrs[i+1]=count;
    // }
    // __syncthreads();
// Call histogram
    // histogram_gpu(A->rowIdxs, result->rowPtrs, A->numRows, A->nnz);

    // Prefix Sum
    // if(threadIdx.x == 0){
    //     unsigned int sum = 0;
    //     for(unsigned int row = 0; row < A->numRows; ++row) {
    //         unsigned int val = result->rowPtrs[row];
    //         result->rowPtrs[row] = sum;
    //         sum += val;
    //     }
    //     result->rowPtrs[A->numRows] = sum;
    // }
   
    // __syncthreads();

    // Binning
    // if(i == 0){
       
    // }

    if( i< A->numRows){
        int col_index =  result->rowPtrs[i];
        int col_index_one = result->rowPtrs[i + 1];
        int size=col_index_one-col_index;

        sort_by_key( &(result->colIdxs[col_index]),&(result->values[col_index]),size);
    }
    // __syncthreads();
    if(i==0){
    A->nnz=0;
    }

}
__global__ void Prefix_sum(CSRMatrix *A){
    thrust::exclusive_scan(thrust::device, A->rowPtrs, A->rowPtrs + A->numRows + 1,A->rowPtrs );
}

__global__ void spmspm(COOMatrix *result, CSRMatrix *A, CSCMatrix *B, float bias) {

    unsigned int r = blockDim.x*blockIdx.x + threadIdx.x;
    unsigned int nnzIdx = 0;
    unsigned int temp=0;

 if (r==0){
    printf("r= %d ",A->nnz);
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
                unsigned int nnzB = B->colPtrs[c + 1] - colPtrB;

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
                            temp = atomicAdd(&result->nnz, 1);
                            result->colIdxs[temp] = c;
                            result->values[temp] = sum;
                            result->rowIdxs[temp] =r ;
                        }
                    }
                }
            }
        }
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
    CSRMatrix *inBuffer  = Y0;
    stopTimeAndPrint(&timer, "Convert feature vectors to CSR");

    // Convert layer weights to CSC
    startTime(&timer);
    CSCMatrix* W[numLayers];
    for(unsigned int layer = 0; layer < numLayers; ++layer) {
        CSCMatrix *W_h = createCSCfromCOO(layerWeights[layer]);
        CSCMatrix W_d;
        W_d.numRows = W_h->numRows;
        W_d.numCols = W_h->numCols;
        W_d.nnz = W_h->nnz;
        W_d.capacity = W_h->capacity;
        cudaMalloc((void **) &W_d.colPtrs, (W_h->numCols + 1) * sizeof(unsigned int));
        cudaMalloc((void **) &W_d.rowIdxs,W_h->nnz * sizeof(unsigned int));
        cudaMalloc((void **) &W_d.values, W_h->nnz * sizeof(float));
        
        cudaMalloc((void **) &W[layer], sizeof(CSCMatrix));
        // CSCMatrix *W_p_d;
        cudaMemcpy(W_d.colPtrs, W_h->colPtrs, (W_h->numCols + 1)* sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(W_d.rowIdxs, W_h->rowIdxs, W_h->nnz * sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(W_d.values, W_h->values, W_h->nnz * sizeof(float), cudaMemcpyHostToDevice);

        cudaMemcpy(W[layer], &W_d, sizeof(CSCMatrix), cudaMemcpyHostToDevice);
    }
    cudaDeviceSynchronize();
    stopTimeAndPrint(&timer, "Convert weights to CSR");


    

    // Allocate memory on GPU
    
    startTime(&timer);
    // ---------- in buffer ------------
    CSRMatrix inBuffer_d;
    inBuffer_d.numRows = inBuffer->numRows;
    inBuffer_d.numCols = inBuffer->numCols;
    inBuffer_d.nnz = inBuffer->nnz;
    inBuffer_d.capacity = inBuffer->numRows*inBuffer->numCols;
    cudaMalloc((void **) &inBuffer_d.rowPtrs, (inBuffer_d.numRows + 1) * sizeof(unsigned int));
    cudaMalloc((void **) &inBuffer_d.colIdxs, inBuffer_d.capacity * sizeof(unsigned int));
    cudaMalloc((void **) &inBuffer_d.values, inBuffer_d.capacity * sizeof(float));

    CSRMatrix *inBuffer_p_d;
    cudaMalloc((void **) &inBuffer_p_d, sizeof(CSRMatrix));


    cudaMemcpy(inBuffer_d.rowPtrs, inBuffer->rowPtrs, (inBuffer->numRows + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(inBuffer_d.colIdxs, inBuffer->colIdxs, inBuffer->nnz * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(inBuffer_d.values, inBuffer->values, inBuffer->nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(inBuffer_p_d, &inBuffer_d, sizeof(CSRMatrix), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopTimeAndPrint(&timer, "Allocate and copy inbuffer");

    startTime(&timer);
    // ----------- out buffer COO --------------
    //emptycoo
    COOMatrix* coo = (COOMatrix*) malloc(sizeof(COOMatrix));
    coo->numRows = inBuffer->numRows;
    coo->numCols = inBuffer->numCols;
    coo->nnz = inBuffer->nnz;
    coo->capacity = inBuffer_d.numRows * inBuffer_d.numCols;
    unsigned int *rowIdxs = (unsigned int *)malloc(inBuffer_d.numRows * inBuffer_d.numCols*sizeof(unsigned int));
    unsigned int *colIdxs = (unsigned int *)malloc(inBuffer_d.numRows * inBuffer_d.numCols*sizeof(unsigned int));
    float *values = (float *)malloc(inBuffer_d.numRows * inBuffer_d.numCols*sizeof(float));
    coo->rowIdxs = rowIdxs;
    coo->colIdxs = colIdxs;
    coo->values = values;

    COOMatrix outBufferCOO_d;
    outBufferCOO_d.numRows = inBuffer_d.numRows;
    outBufferCOO_d.numCols = inBuffer_d.numCols;
    outBufferCOO_d.nnz = 0;
    outBufferCOO_d.capacity = inBuffer_d.numCols * inBuffer_d.numRows;
    cudaMalloc((void **) &outBufferCOO_d.rowIdxs, inBuffer_d.numRows * inBuffer_d.numCols * sizeof(unsigned int));
    cudaMalloc((void **) &outBufferCOO_d.colIdxs, inBuffer_d.numCols * inBuffer_d.numRows * sizeof(unsigned int));
    cudaMalloc((void **) &outBufferCOO_d.values, inBuffer_d.numCols * inBuffer_d.numRows * sizeof(float));
    COOMatrix *outBufferCOO_p_d;
    cudaMalloc((void **) &outBufferCOO_p_d, sizeof(COOMatrix));

    gpuErrchk(cudaMemcpy(outBufferCOO_d.rowIdxs, coo->rowIdxs, coo->capacity * sizeof(unsigned int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(outBufferCOO_d.colIdxs, coo->colIdxs, coo->capacity * sizeof(unsigned int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(outBufferCOO_d.values, coo->values, coo->capacity * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(outBufferCOO_p_d, &outBufferCOO_d, sizeof(COOMatrix), cudaMemcpyHostToDevice));

    cudaDeviceSynchronize();
    stopTimeAndPrint(&timer, "Allocate and copy coo");

    startTime(&timer);

    // ----------- out bufferCSR -----------
    // CSRMatrix *tmp = createEmptyCSR(Y0->numRows, Y0->numCols, Y0->numRows*Y0->numCols);
    // CSRMatrix outBufferCSR_d;
    // outBufferCSR_d.numRows = inBuffer_d.numRows;
    // outBufferCSR_d.numCols = inBuffer_d.numCols;
    // // outBufferCSR_d.nnz = outBuffer->nnz;
    // outBufferCSR_d.capacity = inBuffer_d.capacity;
    // cudaMalloc((void **) &outBufferCSR_d.rowPtrs, (inBuffer_d.numRows + 1) * sizeof(unsigned int));
    // cudaMalloc((void **) &outBufferCSR_d.colIdxs, inBuffer_d.capacity * sizeof(unsigned int));
    // cudaMalloc((void **) &outBufferCSR_d.values, inBuffer_d.capacity * sizeof(float));
    // CSRMatrix *outBufferCSR_p_d;
    // cudaMalloc((void **) &outBufferCSR_p_d, sizeof(CSRMatrix));


    // cudaMemcpy(outBufferCSR_d.rowPtrs, tmp->rowPtrs, (tmp->numRows + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    // cudaMemcpy(outBufferCSR_d.colIdxs, tmp->colIdxs, tmp->capacity * sizeof(unsigned int), cudaMemcpyHostToDevice);
    // cudaMemcpy(outBufferCSR_d.values, tmp->values, tmp->capacity * sizeof(float), cudaMemcpyHostToDevice);
  
    // gpuErrchk(cudaMemcpy(outBufferCSR_p_d, &outBufferCSR_d, sizeof(CSRMatrix), cudaMemcpyHostToDevice));


    // // -------------- W ----------------
    // stopTimeAndPrint(&timer, "Allocate temporary csr");


    // // Copy data from CPU to GPU
    // gpuErrchk(cudaDeviceSynchronize());
    

  


    // Configurations
    const unsigned int threadsPerBlock = BLOCK_DIM;
    const unsigned int blocksPerGrid = (threadsPerBlock + inBuffer->numRows - 1)/threadsPerBlock;


    
    
    // CSRMatrix *t ;
    // cudaMalloc((void **) &t, sizeof(CSRMatrix));
    // Loop over layers
    for(unsigned int layer = 0; layer < numLayers; ++layer) {
        printf("Computing layer %u (SpMSpM)\n", layer);


        // Copy W data to gpu
        // cudaDeviceSynchronize();

        startTime(&timer);
      
        spmspm <<< blocksPerGrid, threadsPerBlock >>>(outBufferCOO_p_d, inBuffer_p_d, W[layer], bias);
        // cudaDeviceSynchronize();
 
        gpuErrchk(cudaDeviceSynchronize());

        stopTimeAndPrint(&timer, "spmspm");

       
        startTime(&timer);
        histogram_gpu<<< blocksPerGrid, threadsPerBlock  >>>( inBuffer_p_d,outBufferCOO_p_d);
        cudaDeviceSynchronize();
        stopTimeAndPrint(&timer, "histogram done");

        startTime(&timer);
        Prefix_sum<<< 1, 1 >>>(inBuffer_p_d);
        cudaDeviceSynchronize();
        stopTimeAndPrint(&timer, "prefix  done");

        startTime(&timer);
        Binning<<< 1, 1 >>>(inBuffer_p_d, outBufferCOO_p_d);
        cudaDeviceSynchronize();
        stopTimeAndPrint(&timer, "binning  done");

        startTime(&timer);
        createCSRfromCOO_gpu <<< blocksPerGrid, threadsPerBlock >>>(inBuffer_p_d, outBufferCOO_p_d);
        cudaDeviceSynchronize();
        stopTimeAndPrint(&timer, "csr convert done");

   

        // Swap buffers
     
       
        // t = inBuffer_p_d;
        // inBuffer_p_d = outBufferCSR_p_d;
        // outBufferCSR_p_d = t;
        // cudaDeviceSynchronize();

    }

    // Copy data from GPU to CPU
    cudaMemcpy(&inBuffer_d, inBuffer_p_d, sizeof(CSRMatrix), cudaMemcpyDeviceToHost);
    cudaMemcpy(inBuffer->rowPtrs, inBuffer_d.rowPtrs, (inBuffer_d.numRows + 1) * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(inBuffer->colIdxs, inBuffer_d.colIdxs, inBuffer_d.nnz * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(inBuffer->values, inBuffer_d.values, inBuffer_d.nnz * sizeof(float), cudaMemcpyDeviceToHost);

    // Free data on GPU
    // ---------- in buffer ------------
    cudaFree(inBuffer_d.rowPtrs);
    cudaFree(inBuffer_d.colIdxs);
    cudaFree(inBuffer_d.values);
    cudaFree(inBuffer_p_d);

    // ----------- out buffer COO -----------
    cudaFree(outBufferCOO_d.rowIdxs);
    cudaFree(outBufferCOO_d.colIdxs);
    cudaFree(outBufferCOO_d.values);
    cudaFree(outBufferCOO_p_d);

    // ----------- out bufferCSR -----------
    // cudaFree(outBufferCSR_d.rowPtrs);
    // cudaFree(outBufferCSR_d.colIdxs);
    // cudaFree(outBufferCSR_d.values);
    // cudaFree(outBufferCSR_p_d);

    // -------------- W ----------------
    // cudaFree(W_d.colPtrs);
    // cudaFree(W_d.rowIdxs);
    // cudaFree(W_d.values);
    // cudaFree(W_p_d);

    // Find nonzero rows
    startTime(&timer);
    findNonzeroRows(result, inBuffer);
    stopTimeAndPrint(&timer, "Find nonzero rows");

    // Free buffers
    startTime(&timer);
    // freeCSR(Y0);
    // for(unsigned int layer = 0; layer < numLayers; ++layer) {
    //     cudaFree(W[layer]);
    //     cudaFree(W[layer]->colPtrs);
    //     cudaFree(W[layer]->rowIdxs);
    //     cudaFree(W[layer]->values);
    // }
    // freeCSR(tmp);
    stopTimeAndPrint(&timer, "Deallocate memory");

}
 