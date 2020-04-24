
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matrix.h"

Vector* createVectorFromFile(const char *fname) {

    // Check if file exists
    FILE* fp = fopen(fname, "r");
    if(fp == NULL) {
        fprintf(stderr, "%s: Error opening file: %s\n", __func__, fname);
        exit(1);
    }

    // Initialize Vector
    Vector* v = createEmptyVector(10000);

    //Read line-by-line and store values
    char *line = NULL;
    size_t len = 0;
    unsigned int nnz = 0;
    while (getline(&line, &len, fp) != -1) {
        if(nnz >= v->capacity) {
            expandVectorCapacity(v, 2*v->capacity);
        }
        sscanf(line, "%d\n", &v->data[nnz]);
        ++nnz;
    }
    v->nnz = nnz;

    // Clean up
    if(line) {
        free(line);
    }
    fclose(fp);

    return v;

}

Vector* createEmptyVector(unsigned int capacity) {
    Vector* v = (Vector*) malloc(sizeof(Vector));
    v->nnz = 0;
    v->capacity = capacity;
    v->data = (unsigned int *)malloc(capacity * sizeof(unsigned int));
    return v;
}

void expandVectorCapacity(Vector* vec, unsigned int capacity) {
    vec->capacity = capacity;
    vec->data = (unsigned int *)realloc(vec->data, capacity * sizeof(unsigned int));
}

void freeVector(Vector* vec) {
    free(vec->data);
    free(vec);
}

void writeVectorToFile(Vector* vec, const char *fname) {
    FILE *fp = fopen(fname, "w");
    if(fp == NULL) {
        fprintf(stderr, "%s: Error while opening the file: %s.\n", __func__, fname);
        exit(1);
    }
    for(unsigned int index = 0; index < vec->nnz; ++index) {
        fprintf(fp,"%d\n", vec->data[index]);
    }
    fclose(fp);
}

COOMatrix* createEmptyCOO(unsigned int numRows, unsigned int numCols, unsigned int capacity) {
    COOMatrix *coo = (COOMatrix *)malloc(sizeof(COOMatrix));
    coo->rowIdxs = (unsigned int *)calloc(1, capacity * sizeof(unsigned int));
    coo->colIdxs = (unsigned int *)malloc( capacity * sizeof(unsigned int));
    coo->values = (float *)malloc( capacity * sizeof(float));
    coo->numRows = numRows;
    coo->numCols = numCols;
    coo->nnz = 0;
    coo->capacity = capacity;
    return coo;
}

COOMatrix* createCOOFromFile(const char *fname, unsigned int maxColumn) {

    // Check if file exists
    FILE *fp = fopen(fname, "r");
    if(fp == NULL) {
        printf("%s: Error opening file: %s\n", __func__, fname);
        exit(1);
    }

    // Initialize COOMatrix arrays
    unsigned int capacity = 10000;
    unsigned int *rowIdxs = (unsigned int *)malloc(capacity*sizeof(unsigned int));
    unsigned int *colIdxs = (unsigned int *)malloc(capacity*sizeof(unsigned int));
    float *values = (float *)malloc(capacity*sizeof(float));

    // Read line-by-line and store values
    char *line = NULL;
    size_t len = 0;
    unsigned int numRows = 0;
    unsigned int nnz = 0;
    while(getline(&line, &len, fp) != -1) {
        unsigned int row, col;
        float val;
        sscanf(line, "%d %d %f\n", &row, &col, &val);
        if(col <= maxColumn) {
            if(nnz >= capacity) {
                capacity = capacity*2;
                rowIdxs = (unsigned int*) realloc(rowIdxs, capacity*sizeof(unsigned int));
                colIdxs = (unsigned int*) realloc(colIdxs, capacity*sizeof(unsigned int));
                values = (float*) realloc(values, capacity*sizeof(float));
            }
            rowIdxs[nnz] = row - 1;
            colIdxs[nnz] = col - 1;
            values[nnz] = val;
            ++nnz;
            if(row > numRows) {
                numRows = row;
            }
        }
    }

    // Free
    if(line) {
        free(line);
    }
    fclose(fp);

    COOMatrix* coo = (COOMatrix*) malloc(sizeof(COOMatrix));
    coo->numRows = numRows;
    coo->numCols = maxColumn;
    coo->nnz = nnz;
    coo->capacity = capacity;
    coo->rowIdxs = rowIdxs;
    coo->colIdxs = colIdxs;
    coo->values = values;

    return coo;
}

void expandCOOCapacity(COOMatrix* A, unsigned int capacity) {
    A->capacity = capacity;
    A->rowIdxs = (unsigned int *)realloc(A->rowIdxs, capacity * sizeof(unsigned int));
    A->colIdxs = (unsigned int *)realloc(A->colIdxs, capacity * sizeof(unsigned int));
    A->values = (float *)realloc(A->values, capacity * sizeof(float));
    if( A->rowIdxs==NULL || A->colIdxs==NULL || A->values==NULL ) {
        printf("%s: Error allocating memory.\n", __func__);
        exit(1);
    }
}

void freeCOO(COOMatrix* coo) {
    free(coo->rowIdxs);
    free(coo->colIdxs);
    free(coo->values);
    free(coo);
}

void writeCOOtoFile(COOMatrix* A, const char *fname) {
    FILE *fp = fopen(fname, "w");
    if(fp == NULL) {
        fprintf(stderr, "%s: Error while opening the file: %s.\n", __func__, fname);
        exit(1);
    }
    for(unsigned int index = 0; index < A->nnz; ++index) {
        fprintf(fp,"%u %u %f\n", A->rowIdxs[index], A->colIdxs[index], A->values[index]);
    }
    fclose(fp);
}

CSRMatrix* createEmptyCSR(unsigned int numRows, unsigned int numCols, unsigned int capacity) {
    CSRMatrix *csr = (CSRMatrix *)malloc(sizeof(CSRMatrix));
    csr->rowPtrs= (unsigned int *)calloc(1, (numRows+1) * sizeof(unsigned int));
    csr->colIdxs= (unsigned int *)malloc( capacity * sizeof(unsigned int));
    csr->values= (float *)malloc( capacity * sizeof(float));
    csr->numRows = numRows;
    csr->numCols = numCols;
    csr->nnz = 0;
    csr->capacity = capacity;
    return csr;
}

void quicksort(float *data, unsigned int *key, unsigned int start, unsigned int end) {
    if((end - start + 1) > 1) {
        unsigned int left = start, right = end;
        unsigned int pivot = key[right];
        while(left <= right) {
            while(key[left] < pivot) {
                left = left + 1;
            }
            while(key[right] > pivot) {
                right = right - 1;
            }
            if(left <= right) {
                unsigned int tmpKey = key[left]; key[left] = key[right]; key[right] = tmpKey;
                float tmpData = data[left]; data[left] = data[right]; data[right] = tmpData;
                left = left + 1;
                right = right - 1;
            }
        }
        quicksort(data, key, start, right);
        quicksort(data, key, left, end);
    }
}

void convertCOOtoCSR(COOMatrix* A, CSRMatrix* B) {

    // Check compatibility
    if(B->numRows != A->numRows || B->numCols != A->numCols) {
        fprintf(stderr, "%s: matrices have incompatible dimensions!\n", __func__);
        exit(1);
    }
    if(B->capacity < A->nnz) {
        fprintf(stderr, "%s: CSR matrix has insufficient capacity!\n", __func__);
        exit(1);
    }

    // Set nonzeros
    B->nnz = A->nnz;

    // Histogram
    memset(B->rowPtrs, 0, (B->numRows + 1)*sizeof(unsigned int));
    for(unsigned int i = 0; i < A->nnz; ++i) {
        unsigned int row = A->rowIdxs[i];
        B->rowPtrs[row]++;
    }

    // Prefix sum
    unsigned int sum = 0;
    for(unsigned int row = 0; row < A->numRows; ++row) {
        unsigned int val = B->rowPtrs[row];
        B->rowPtrs[row] = sum;
        sum += val;
    }
    B->rowPtrs[A->numRows] = sum;

    // Binning
    for(unsigned int index = 0; index < A->nnz; ++index) {
        unsigned int row = A->rowIdxs[index];
        unsigned int i = B->rowPtrs[row]++;
        B->colIdxs[i] = A->colIdxs[index];
        B->values[i] = A->values[index];
    }

    // Restore row pointers
    for(unsigned int row = A->numRows - 1; row > 0; --row) {
        B->rowPtrs[row] = B->rowPtrs[row - 1];
    }
    B->rowPtrs[0] = 0;

    // Sort nonzeros within each row
    for(unsigned int row = 0; row < B->numRows; ++row) {
        unsigned int start = B->rowPtrs[row];
        unsigned int end = B->rowPtrs[row + 1] - 1;
        quicksort(B->values, B->colIdxs, start, end);
    }

}

void expandCSRCapacity(CSRMatrix* A, unsigned int capacity) {
    A->capacity = capacity;
    A->colIdxs = (unsigned int *)realloc(A->colIdxs, capacity * sizeof(unsigned int));
    A->values = (float *)realloc(A->values, capacity * sizeof(float));
    if( A->colIdxs==NULL || A->values==NULL ) {
        printf("%s: Error allocating memory.\n", __func__);
        exit(1);
    }
}

void freeCSR(CSRMatrix* csr) {
    free(csr->rowPtrs);
    free(csr->colIdxs);
    free(csr->values);
    free(csr);
}

void writeCSRtoFile(CSRMatrix* A, const char *fname) {
    FILE *fp = fopen(fname, "w");
    if(fp == NULL) {
        fprintf(stderr, "%s: Error while opening the file: %s.\n", __func__, fname);
        exit(1);
    }
    for(unsigned int r = 0; r < A->numRows; ++r) {
        for(unsigned int index = A->rowPtrs[r]; index < A->rowPtrs[r + 1]; ++index) {
            fprintf(fp,"%u %u %f\n", r, A->colIdxs[index], A->values[index]);
        }
    }
    fclose(fp);
}

CSCMatrix* createCSCfromCOO(COOMatrix* A) {

    // Allocate
    unsigned int *colPtrs= (unsigned int *) calloc(A->numCols + 1, sizeof(unsigned int));
    unsigned int *rowIdxs = (unsigned int *) malloc(A->nnz*sizeof(unsigned int));
    float *values = (float *) malloc(A->nnz*sizeof(float));
    
    // Histogram
    for(unsigned int i = 0; i < A->nnz; ++i) {
        unsigned int col = A->colIdxs[i];
        colPtrs[col]++;
    }

    // Prefix sum
    unsigned int sum = 0;
    for(unsigned int col = 0; col < A->numCols; ++col) {
        unsigned int val = colPtrs[col];
        colPtrs[col] = sum;
        sum += val;
    }
    colPtrs[A->numCols] = sum;

    // Binning
    for(unsigned int index = 0; index < A->nnz; ++index) {
        unsigned int col = A->colIdxs[index];
        unsigned int i = colPtrs[col]++;
        rowIdxs[i] = A->rowIdxs[index];
        values[i] = A->values[index];
    }

    // Restore column pointers
    for(unsigned int col = A->numCols - 1; col > 0; --col) {
        colPtrs[col] = colPtrs[col - 1];
    }
    colPtrs[0] = 0;
    
    CSCMatrix* csc = (CSCMatrix*) malloc(sizeof(CSCMatrix));
    csc->colPtrs = colPtrs;
    csc->rowIdxs = rowIdxs;
    csc->values = values;
    csc->nnz = A->nnz;
    csc->numRows = A->numRows;
    csc->numCols = A->numCols;
    csc->capacity = A->nnz;

    return csc;

}

void freeCSC(CSCMatrix* csc) {
    free(csc->colPtrs);
    free(csc->rowIdxs);  
    free(csc->values);
    free(csc);
}

