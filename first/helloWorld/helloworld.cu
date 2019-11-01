#include <stdio.h>

__global__ void helloFromGPU()
{
	printf("Hello from GPU! BlockID: %d - ThreadID: %d.\n", blockIdx.x, threadIdx.x);
}

int main()
{
	// Hello from CPU. 
	printf("Hello from CPU!\n");

	helloFromGPU<<<2, 5>>>();

	cudaDeviceReset();
	return 0;
}


