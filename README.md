# Sparse-Neural-Networks-GPU
GPU Computing course project

This project aims to accelerate the evaluation of sparse neural networks.

## Description of the sequential code:
   The sequential code starts by converting featureVectors to CSR and the converting layer weights to CSC.The code loops over every layer ,and does spmspm multiplication(explained bellow) taking advantage of double buffering.After that it  checks for non zero rows and store them into result vector.
    ### The SPMSPM multiplication: 
        The SPMSPM multiplication  happens by looping over non empty raws of feature vectors.Then
        for every non empty raw loop over all non empty column and the calculate the dot product by 
        finding the intersction.And finally storing postive results in a CSR Form  matrix.


## Compiling the code:
    * Bullet list Compile all versions: make
    * Compile a specific version: make spnn_<version>
    * Where <version> can be cpu, gpu0, gpu1, gpu2, or gpu3
## Running the code: ./spnn_< version >

    By default, the following configurations are used:
        * #layers: 120
        * #neurons/layer : 1024
        * Directory for reading input data: data
        * You can provide your own configurations as follows:
             ./spnn_<version> -d <input directory> -l <#layers> -n <#neurons/layer>

