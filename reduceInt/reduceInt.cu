#include <cuda_runtime.h>
#include <stdio.h>

#include <sys/time.h>

double cpuSecond()
{   
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.0e-6);
}

int recursiveReduce(int *input, const int size)
{
	int sum = 0;
	for (int i=0; i<size; i++)
	{
		sum += input[i];
	}

	return sum;
}

__global__  void reduceNeighbored(int *g_idata, int *g_odata, unsigned int n)
{
	// set thread id.
	unsigned int tid = threadIdx.x;

	// convert global data pointer to th local pointer of this block. 
	int *idata = g_idata + blockIdx.x * blockDim.x;

	// boundary check.
	//if (idx >= n)
	//	return;

	// in-place reduction in global memory. 
	for (int stride = 1; stride < blockDim.x; stride *= 2)
	{
		if ( (tid % (2 * stride)) == 0)
		{
			idata[tid] += idata[tid + stride];
		}

		// synchronize within block. 
		__syncthreads();
	}

	// write result for this block to global mem. 
	if (tid == 0)
	{
		g_odata[blockIdx.x] = idata[0];
	}
}

int main(int argc, char **argv)
{
	// set up device. 
	int dev = 0;

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	printf("%s starting reduction at \n", argv[0]);
	printf("Device %d: %s \n", dev, deviceProp.name);
	cudaSetDevice(dev);

	bool bResult = false;
	
	// initialization. 
	int size = 1 << 24; 
	printf("With array size: (%d), ", size);

	// execution configuration. 
	int blocksize = 512;
	if (argc > 1)
	{
		blocksize = atoi(argv[1]);
	}

	dim3 block (blocksize, 1);
	dim3 grid ( (size + block.x -1)/block.x, 1);
	printf("grid: (%d), block: (%d). \n", grid.x, block.x);

	// allocate host memory. 
	size_t bytes = size * sizeof(int);
	int *h_idata = (int *)malloc(bytes);
	int *h_odata = (int *)malloc(grid.x * sizeof(int));
	int *tmp = (int *)malloc(bytes);

	// initialize the array. 
	for (int i =0; i<size; i++)
	{
		// mask off high 2 bytes to force max number to 255. 
		h_idata[i] = (int)(rand() & 0xFF);
	}

	memcpy(tmp, h_idata, bytes);

	double iStart, iElaps;
	int gpu_sum = 0;

	// allocate device memory. 
	int *d_idata = NULL;
	int *d_odata = NULL;

	cudaMalloc((void **) &d_idata, bytes);
	cudaMalloc((void **) &d_odata, grid.x * sizeof(int));

	// cpu reduction.
	iStart = cpuSecond();
	int cpu_sum = recursiveReduce(tmp, size);
	iElaps = cpuSecond() - iStart;
	printf("cpu reduce elapsed %lfs cpu seconds, sum value: %d\n", iElaps, cpu_sum);
	


	return 0;
}







