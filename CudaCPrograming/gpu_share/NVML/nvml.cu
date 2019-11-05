#include <cuda_runtime.h>
#include <nvml.h>
#include <iostream>
#include <iomanip>
#include <string> 

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

    auto return_code = nvmlInit();
    if (return_code != NVML_SUCCESS)
    {
        std::cout << "nvmlInit returned: " 
                  << (int)return_code << " -> " 
                  << nvmlErrorString(return_code) << std::endl; 
        return -1; 
    }

    nvmlDevice_t device; 
    return_code = nvmlDeviceGetHandleByIndex(0, &device);
    if (return_code != NVML_SUCCESS)
    {
        std::cout << "nvmlDeviceGetHandleByIndex returned: " 
                  << (int)return_code << " -> " 
                  << nvmlErrorString(return_code) << std::endl; 
        return -1; 
    }
    
    nvmlMemory_t memory; 
    return_code = nvmlDeviceGetMemoryInfo(device, &memory);
    if (return_code != NVML_SUCCESS)
    {
        std::cout << "nvmlDeviceGetHandleByIndex returned: " 
                  << (int)return_code << " -> " 
                  << nvmlErrorString(return_code) << std::endl; 
        return -1; 
    }
    std::cout << "Total Mem  : " << memory.total/1000000 << " MB\n"
              << "Used Mem   : " << memory.used/1000000 << " MB\n"
              << "Free Mem   : " << memory.free/1000000 << " MB" << std::endl; 
    
    nvmlProcessInfo_t* p_infos = NULL; 
    unsigned int p_count = 0;
    // First check how many processes are running in the kernel. 
    return_code = nvmlDeviceGetComputeRunningProcesses(device, &p_count, p_infos);
    if (return_code != NVML_SUCCESS && return_code != NVML_ERROR_INSUFFICIENT_SIZE)
    {
        std::cout << "nvmlDeviceGetComputeRunningProcesses returned: " 
                  << (int)return_code << " -> " 
                  << nvmlErrorString(return_code) << std::endl; 
        return -1; 
    }
    std::cout << "There are " << p_count << " processes runing. " << std::endl; 

    // Allocate space and get the detailed infomation. 
    if (p_count>0)
    {
        p_infos = (nvmlProcessInfo_t*)malloc(sizeof(nvmlProcessInfo_t)*(p_count+10));
        return_code = nvmlDeviceGetComputeRunningProcesses(device, &p_count, p_infos);
        if (return_code != NVML_SUCCESS)
        {
            std::cout << "nvmlDeviceGetComputeRunningProcesses returned: " 
                    << (int)return_code << " -> " 
                    << nvmlErrorString(return_code) << std::endl; 
            return -1; 
        }

        char process_name[50];
        memset(process_name, '\0', sizeof(char)*50);  
        std::cout.setf(std::ios::left);
        std::cout << std::setw(8) << "PID" << std::setw(50) << "Process Name"
                  << std::setw(19) << "GPU MEM" << std::endl; 
        for (unsigned i=0; i<p_count; ++i)
        {
            nvmlSystemGetProcessName(p_infos[i].pid, process_name, 50);
            std::cout << std::setw(8) << p_infos[i].pid
                      << std::setw(50) << process_name  
                      << p_infos[i].usedGpuMemory/1000000 << " MB" << std::endl; 
            memset(process_name, '\0', sizeof(char)*50);
        }
        std::cout << std::endl; 

        free(p_infos);
    }
    nvmlShutdown();
    return 0; 
}