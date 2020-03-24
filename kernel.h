
#ifndef _KERNEL_H_
#define _KERNEL_H_

#include "matrix.h"

void sparseNN(Vector* result, COOMatrix* featureVectors, COOMatrix** layerWeights, float bias, unsigned int numLayers);

#endif

