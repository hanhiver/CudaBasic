#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

__global__ void gpu_sleep(const int sleep_time)
{   
    int tmp = 0;
    for (int i=sleep_time; i<sleep_time; i++)
        tmp += i;

    printf("GPU job threadId (%d) done, sleep for %d seconds.\n", threadIdx.x, sleep_time);
}


int main(int argc, char **argv)
{
    // set up device. 
    int dev_count;
    int dev = 0;
    cudaDeviceProp dprop;
    CHECK(cudaGetDeviceCount(&dev_count));
    CHECK(cudaGetDeviceProperties(&dprop, dev));
    printf("There are %d devices in the system. \n", dev_count);
    printf("%s start at device %d: %s \n", argv[0], dev, dprop.name);
    CHECK(cudaSetDevice(dev));
    
    int sleep_time = 1;
    if (argc > 1)
    {
        sleep_time = atoi(argv[1]);
    }

    int blocksize = 1;
    if (argc > 2)
    {
        blocksize = atoi(argv[2]);
    }
        
    // execution configuration
    dim3 block (blocksize);
    dim3 grid (1);

    // kernel: sleep.
    gpu_sleep <<<grid, block>>> (sleep_time);
    
    sleep(sleep_time);

    // reset device.
    CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}

