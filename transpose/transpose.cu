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

void verifyResult(int *A, int *B, const int size)
{
    bool match = 1; 

    for (int i=0; i<size; i++)
    {
        if (A[i] != B[i])
        {
            match = 0;
            printf("Not match in pos: %d, (%d != %d). \n", i, A[i], B[i]);
            break;
        }
    } 

    if (!match)
    {
        printf("Test failed! \n");
    }
    else
    {
        printf("Test OK! \n");
    }
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

// warmup kernel. 
__global__ void warmup(int *in, int *out, const int nx, const int ny)
{
    // set thread id.
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;

	if (ix < nx && iy < ny)
	{
		out[iy * nx + ix] = in[ix * ny + iy];
	}
}

// kernel 0: access data in row.  
__global__ void copyRow(int *in, int *out, const int nx, const int ny)
{
    // set thread id.
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;

	if (ix < nx && iy < ny)
	{
		out[iy * nx + ix] = in[iy * nx + ix];
	}
}

// kernel 1: access data in col. 
__global__ void copyCol(int *in, int *out, const int nx, const int ny)
{
    // set thread id.
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;

	if (ix < nx && iy < ny)
	{
		out[ix * ny + iy] = in[ix * ny + iy];
	}
}

// kernel 2: read data in row and write in col  
__global__ void transposeNaiveRow(int *in, int *out, const int nx, const int ny)
{
    // set thread id.
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;

	if (ix < nx && iy < ny)
	{
		out[ix * ny + iy] = in[iy * nx + ix];
	}
}

// kernel 3: read data in col and write in row. 
__global__ void transposeNaiveCol(int *in, int *out, const int nx, const int ny)
{
    // set thread id.
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;

	if (ix < nx && iy < ny)
	{
		out[iy * nx + ix] = in[ix * ny + iy];
	}
}

// kernel 4: read data in row and write in col + unroll 4
__global__ void transposeUnroll4Row(int *in, int *out, const int nx, const int ny)
{
    // set thread id.
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x * 4;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;

    unsigned int ti = iy * nx + ix; // access in rows. 
    unsigned int to = ix * ny + iy; // access in cols. 

	if (ix + 3 * blockDim.x < nx && iy < ny)
	{
		out[to]                       = in[ti];
        out[to + ny * blockDim.x]     = in[ti + blockDim.x];
        out[to + ny * blockDim.x * 2] = in[ti + blockDim.x * 2];
        out[to + ny * blockDim.x * 3] = in[ti + blockDim.x * 3];
	}
}

// kernel 5: read data in row and write in col + unroll 4
__global__ void transposeUnroll4Col(int *in, int *out, const int nx, const int ny)
{
    // set thread id.
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x * 4;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;

    unsigned int ti = iy * nx + ix; // access in rows. 
    unsigned int to = ix * ny + iy; // access in cols. 

	if (ix + 3 * blockDim.x < nx && iy < ny)
	{
		out[ti]                  = in[to];
        out[ti + blockDim.x]     = in[to + ny * blockDim.x];
        out[ti + blockDim.x * 2] = in[to + ny * blockDim.x * 2];
        out[ti + blockDim.x * 3] = in[to + ny * blockDim.x * 3];
	}
}


int main(int argc, char **argv)
{
    int ikernel = 0;
    
    if (argc > 1) ikernel = atoi(argv[1]);

	int nx = 1<<12; 
	int ny = 1<<12;
	int nxy = nx * ny; 
	size_t nbytes = nxy * sizeof(int);

	int *src = (int *)malloc(nbytes);
	int *dst = (int *)malloc(nbytes);
    int *refGpu = (int *)malloc(nbytes);
	
	memset(src, 0, nxy);
	memset(dst, 0, nxy);

	initialMatrix(src, nxy);
	//printMatrix(src, nx, ny);
    transposeHost(src, dst, nx, ny);
    //printMatrix(dst, ny, nx);

	// set execution configuration. 
	int blockx = 32; 
    int blocky = 32;
    
    if (argc>2) blockx = atoi(argv[2]);
    if (argc>3) blocky = atoi(argv[3]);

    dim3 block(blockx, blocky);
	dim3 grid( (nx+block.x-1)/block.x, (ny+block.y-1)/block.y );
	
	// allocate memory. 
	int *d_src, *d_dst;
	CHECK(cudaMalloc((int **) &d_src, nbytes));
	CHECK(cudaMalloc((int **) &d_dst, nbytes));

    // warm up to avoid startup overhead. 
    double iStart = seconds();
    warmup <<<grid, block>>> (d_src, d_dst, nx, ny);
    double iElaps = seconds() - iStart;
    printf("warmup             elapsed %f sec\n", iElaps);
    CHECK(cudaGetLastError());

    // copy the data from host to device. 
    CHECK(cudaMemcpy(d_src, src, nbytes, cudaMemcpyHostToDevice));

    void (*kernel)(int *, int *, int, int);
    const char *kernelName; 
    bool verify = 1; 

    // set up kernel. 
    switch (ikernel)
    {
    case 0:
        kernel = &copyRow;
        kernelName = "copyRow             ";
        verify = 0;
        break;

    case 1:
        kernel = &copyCol;
        kernelName = "copyCol             ";
        verify = 0;
        break;

    case 2:
        kernel = &transposeNaiveRow;
        kernelName = "transposeNaiveRow   ";
        break;
    
    case 3:
        kernel = &transposeNaiveCol;
        kernelName = "transposeNaiveCol   ";
        break;
    
    case 4:
        kernel = &transposeUnroll4Row;
        kernelName = "transposeUnroll4Row ";
        grid.x = (nx + block.x * 4 -1) / (block.x * 4);
        break;
    
    case 5:
        kernel = &transposeUnroll4Col;
        kernelName = "transposeUnroll4Col ";
        grid.x = (nx + block.x * 4 -1) / (block.x * 4);
        break;
    
    default:
        printf("Wrong kernel index.\n");
        exit(-1);
    }
    
    iStart = seconds();
    kernel <<<grid, block>>> (d_src, d_dst, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;

    // caculate effective_bandwidth.
    float ibnd = 2 * nbytes / 1e9 / iElaps;
    printf("%s elapsed %f sec <<<grid (%d, %d), block (%d, %d)>>> effective bandwidth %f GB\n", 
            kernelName, iElaps, grid.x, grid.y, block.x, block.y, ibnd);

    // copy the result from device to host.
    if (verify)
    {
        CHECK(cudaMemcpy(refGpu, d_dst, nbytes, cudaMemcpyDeviceToHost));
        verifyResult(dst, refGpu, nxy);
    }

    // free the device memory.
    cudaFree(d_src);
    cudaFree(d_dst);

    // free the host memory.
    free(src);
    free(dst);
    free(refGpu);

	return EXIT_SUCCESS;
}


