
#include <stdlib.h>
#include <stdio.h>

#include "verify.h"

void verify(Vector* scores, char* categoriesFileName) {

    Vector* trueCategories = createVectorFromFile(categoriesFileName);

    unsigned int passed = 1;
    if(scores->nnz != trueCategories->nnz) {
        passed = 0;
    } else {
        for(unsigned int i=0; i<scores->nnz; ++i) {
            if(scores->data[i] != trueCategories->data[i] - 1) { //Row values stored in file are 1-based, not 0-based
                passed = 0;
                break;
            }
        }
    }

    if(passed) {
        printf("Challenge PASSED\n");
    } else {
        printf("Challenge FAILED\n");
    }

    freeVector(trueCategories);

}

