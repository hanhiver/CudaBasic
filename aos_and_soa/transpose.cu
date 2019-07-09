#include <cuda_runtime.h>
#include <stdio.h>

#include "../common.h"

//#define nx 3;
//#define ny 2;

void initialMatrix(int *data, const int size)
{
	for (int i=0; i<size; i++)
	{
		data[i] = i;
	}
}

void printMatrix(int *data, const int nx, const int ny)
{
    printf("\n");
    for (int iy=0; iy<ny; iy++)
    {
        for (int ix=0; ix<nx; ix++)
        {
            printf("%d\t", data[iy*nx+ix]);
        }
        printf("\n");
    }
    printf("\n");
}

void transposeHost(int *in, int *out, const int nx, const int ny)
{
	for (int iy=0; iy<ny; iy++)
	{
		for (int ix=0; ix<nx; ix++)
		{
			out[ix*ny + iy] = in[iy*nx + ix];
		}
	}
}

__global__ void transposeGpu(int *in, int *out, const int nx, const int ny)
{
    // set thread id.
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // convert global data point to the local pointer of this block. 
    int *idata = g_idata + blockIdx.x * blockDim.x;

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
	printMatrix(src, nx, ny);
    transposeHost(src, dst, nx, ny);
    printMatrix(dst, ny, nx);

	return EXIT_SUCCESS;
}


