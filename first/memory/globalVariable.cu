#include <cuda_runtime.h>
#include <stdio.h>

#include "../common.h"

__device__ float devData;

__global__ void checkGlobalVariable()
{
    // display the original value.
    printf("Device: the value is %f\n", devData);

    // alter the value.
    devData *= 2.0f;
}

int main(void)
{
    // initialize the global variable. 
    float value = 3.14f;

    CHECK(cudaMemcpyToSymbol(devData, &value, sizeof(float)));
    printf("Host: copied %f to the global variable.\n", value);

    // invoke the kernel. 
    checkGlobalVariable <<<2, 2>>>();

    // copy the global variable back to the host. 
    CHECK(cudaMemcpyFromSymbol(&value, devData, sizeof(float)));
    printf("Host: the value changed by the kernel to %f\n", value);

    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}


