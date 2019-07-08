#include <cuda_runtime.h>
#include <stdio.h>

#include "../common.h"

//#define nx 3;
//#define ny 2;

void initialMatrix(int *data, const int size)
{
	for (int i=0; i<size; i++)
	{
		data[i] = rand();
	}
}

void transposeHost(int *in, int *out, const int nx, const int ny)
{
	for (int i=0; i<nx; i++)
	{
		for (int j=0; j<ny; j++)
		{
			out[i*nx + j] = in[j*ny + i];
		}
	}
}

__global__ void transposeGpu(int *in, int *out, const int nx, const int ny)
{

}

int main(int argc, char **argv)
{
	int nx = 3; 
	int ny = 2;
	int nxy = nx * ny; 
	size_t bytes = nxy * sizeof(int);

	int *src = (int *)malloc(bytes);
	int *dst = (int *)malloc(bytes);
	
	memset(src, 0, nxy);

	initialMatrix(src, nxy);
	
	for (int i=0; i<nxy; i++)
	{
		printf("%d\t", src[i]);
	}
	printf("\n");

	return EXIT_SUCCESS;
}


