
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

COOMatrix* createEmptyCOO(unsigned int numRows, unsigned int numCols, unsigned int capacity);
COOMatrix* createCOOFromFile(const char *fname, unsigned int maxColumn);
void expandCOOCapacity(COOMatrix* A, unsigned int capacity);
void freeCOO(COOMatrix* coo);
void writeCOOtoFile(COOMatrix* A, const char *fname);

typedef struct CSRMatrix {
    unsigned int numRows;
    unsigned int numCols;
    unsigned int nnz;
    unsigned int capacity;
    unsigned int* rowPtrs;
    unsigned int* colIdxs;
    float* values;
} CSRMatrix;

CSRMatrix* createEmptyCSR(unsigned int numRows, unsigned int numCols, unsigned int capacity);
void convertCOOtoCSR(COOMatrix* A, CSRMatrix* B);
void expandCSRCapacity(CSRMatrix* A, unsigned int capacity);
void freeCSR(CSRMatrix* csr);
void writeCSRtoFile(CSRMatrix* A, const char *fname);

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

