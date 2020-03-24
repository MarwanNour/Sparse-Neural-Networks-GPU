
#ifndef _MATRIX_H_
#define _MATRIX_H_

typedef struct Vector {
    unsigned int nnz;
    unsigned int capacity;
    unsigned int *data;
} Vector;

Vector* createVectorFromFile(const char *fname);
Vector* createEmptyVector(unsigned int capacity);
void expandVectorCapacity(Vector* vec, unsigned int capacity);
void freeVector(Vector* vec);
void writeVectorToFile(Vector* vec, const char *fname);

typedef struct COOMatrix {
    unsigned int numRows;
    unsigned int numCols;
    unsigned int nnz;
    unsigned int capacity;
    unsigned int* rowIdxs;
    unsigned int* colIdxs;
    float* values;
} COOMatrix;

COOMatrix* createCOOFromFile(const char *fname, unsigned int maxColumn);
void freeCOO(COOMatrix* coo);

typedef struct CSRMatrix {
    unsigned int numRows;
    unsigned int numCols;
    unsigned int nnz;
    unsigned int capacity;
    unsigned int* rowPtrs;
    unsigned int* colIdxs;
    float* values;
} CSRMatrix;

CSRMatrix* createCSRfromCOO(COOMatrix* A);
CSRMatrix* createEmptyCSR(unsigned int numRows, unsigned int numCols, unsigned int capacity);
void expandCSRCapacity(CSRMatrix* A, unsigned int capacity);
void freeCSR(CSRMatrix* csr);

typedef struct CSCMatrix {
    unsigned int numRows;
    unsigned int numCols;
    unsigned int nnz;
    unsigned int capacity;
    unsigned int* colPtrs;
    unsigned int* rowIdxs;
    float* values;
} CSCMatrix;

CSCMatrix* createCSCfromCOO(COOMatrix* A);
void freeCSC(CSCMatrix* csc);

#endif

