#include <cuda_runtime.h>
#include <stdio.h>

#include "../common.h"

__global__ void nestedHelloWorld(int const iSize, int iDepth)
{
    int tid = threadIdx.x;
    printf("Recursion = %d: \"Hello, World!\" from thread %d, block %d\n", iDepth, tid, blockIdx.x);

    // condision to stop recursive execution. 
    if (iSize == 1) return;

    // reduce block size to half.
    int nthreads = iSize >> 1;

    // thread 0 launches child grid recursively. 
    if (tid == 0 && nthreads > 0)
    {
        nestedHelloWorld<<<1, nthreads>>>(nthreads, ++iDepth);
        printf("------> nested execution depth: %d\n", iDepth);
    }
}

int main(int argc, char **argv)
{
    int size = 8;
    int blocksize = 8;
    int igrid = 1;

    if (argc > 1)
    {
        igrid = atoi(argv[1]);
        size = blocksize * igrid;
    }

    printf("%s Starting with size: %d, blocksize: %d, grid: %d.\n", argv[0], size, blocksize, igrid);

    dim3 block (blocksize);
    dim3 grid ((size + block.x -1)/block.x, 1);

    nestedHelloWorld<<<grid, block>>>(block.x, 0);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}


