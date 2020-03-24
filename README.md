# Sparse-Neural-Networks-GPU
GPU Computing course project

This project aims to accelerate the evaluation of sparse neural networks.


## Compiling the code:
* Compile all versions: make
* Compile a specific version: make spnn_<version>
   * Where <version> can be cpu, gpu0, gpu1, gpu2, or gpu3
## Running the code: ./spnn_<version>

By default, the following configurations are used:
* #layers: 120
* #neurons/layer : 1024
* Directory for reading input data: data
* You can provide your own configurations as follows:
./spnn_<version> -d <input directory> -l <#layers> -n <#neurons/layer>
