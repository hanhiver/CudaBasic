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

// kernel 1: ready by col and write by row. 
__global__ void transposeGpu(int *in, int *out, const int nx, const int ny)
{
    // set thread id.
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;

	if (idx < nx && idy < ny)
	{
		out[idy * nx + idx] = in[idx * ny + idy];
	}
}

int main(int argc, char **argv)
{
	int nx = 1<<20; 
	int ny = 1<<20;
	int nxy = nx * ny; 
	size_t bytes = nxy * sizeof(int);

	int *src = (int *)malloc(bytes);
	int *dst = (int *)malloc(bytes);
	
	memset(src, 0, nxy);
	memset(dst, 0, nxy);

	initialMatrix(src, nxy);
	//printMatrix(src, nx, ny);
    transposeHost(src, dst, nx, ny);
    //printMatrix(dst, ny, nx);

	// set execution configuration. 
	dim3 block(32, 32);
	dim3 grid( (nx+block.x-1)/block.x, (ny+block.y-1)/block.y );
	
	// allocate memory. 
	int *d_src, *d_dst;
	CHECK(cudaMalloc((int **) &d_src, bytes));
	CHECK(cudaMalloc((int **) &d_dst, bytes));

	return EXIT_SUCCESS;
}


