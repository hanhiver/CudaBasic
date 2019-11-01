#include <stdio.h> 

__global__ void helloFromGPU(void)
{
    printf("Hello from GPU.\n");
}

int main()
{
    printf("Hello from CPU.\n");

    helloFromGPU<<<2, 5>>>(); 
    cudaDeviceReset();

    return 0; 
}