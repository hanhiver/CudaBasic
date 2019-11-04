#include <cuda_runtime.h>
#include <iostream>

int main()
{
    std::cout << "Starting to check the GPU info. " << std::endl; 

    int deviceCount = 0; 
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
        std::cout << "cudaGetDeviceCount returned: " 
                  << (int)error_id << " -> " 
                  << cudaGetErrorString(error_id) << std::endl; 
        return -1; 
    }

    std::cout << "There are " << deviceCount << " CUDA device(s) in the system.\n";

    return 0; 
}