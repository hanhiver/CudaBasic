#include <cuda_runtime.h>
#include <iostream>

int main()
{
    std::cout << "Starting to check the GPU info. " << std::endl; 

    int deviceCount = 0; 
    //cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    auto error_id = cudaGetDeviceCount(&deviceCount);
    if (error_id != cudaSuccess)
    {
        std::cout << "cudaGetDeviceCount returned: " 
                  << (int)error_id << " -> " 
                  << cudaGetErrorString(error_id) << std::endl; 
        return -1; 
    }
    std::cout << "There are " << deviceCount << " CUDA device(s) in the system.\n";

    nvmlDevice_t device; 
    error_id = nvmlDeviceGetHandleByIndex(0, &device);
    if (error_id != cudaSuccess)
    {
        std::cout << "nvmlDeviceGetHandleByIndex returned: " 
                  << (int)error_id << " -> " 
                  << cudaGetErrorString(error_id) << std::endl; 
        return -1; 
    }
    
    nvmlMemory_t memory; 
    error_id = nvmlDeviceGetMemoryInfo(device, &memory);
    if (error_id != cudaSuccess)
    {
        std::cout << "nvmlDeviceGetHandleByIndex returned: " 
                  << (int)error_id << " -> " 
                  << cudaGetErrorString(error_id) << std::endl; 
        return -1; 
    }
    std::cout << "Total Mem  : " << memory.total << "\n"
              << "Used Mem   : " << memory.used << "\n"
              << "Free Mem   : " << memory.free << std::endl; 

    return 0; 
}