#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> 
#include <time.h>

__global__ void mk_kernel(char* keep_mem, size_t bytes)
{
    for (unsigned i=0; i<bytes; ++i)
    {
        keep_mem[i] = 0; 
    }
}

int main()
{
    unsigned bytes = 1024 * 1024 * 1024; 
    printf("I will sleep for 5 seconds. \n");

    char *keep_mem; 
    cudaMalloc(&keep_mem, sizeof(char)*bytes);
    
    //mk_kernel<<<1, 1>>>(keep_mem, bytes);

    sleep(5);
    printf("Done. \n");

    cudaFree(keep_mem);
    cudaDeviceReset();

    return 0; 
}