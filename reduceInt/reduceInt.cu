#include <cuda_runtime.h>
#include <stdio.h>

#include <sys/time.h>
#include "../common.h"

int normalReduce(int *input, const int size)
{
	int sum = 0;
	for (int i=0; i<size; i++)
	{
		sum += input[i];
	}

	return sum;
}

int recursiveReduce(int *data, const int size)
{
    // terminate check. 
    if (size == 1)
        return data[0];

    // renew the stride
    int const stride = size / 2;

    // in-place reduction. 
    for (int i=0; i<stride; i++)
    {
        data[i] += data[i + stride];
    }

    // call recursively. 
    return recursiveReduce(data, stride);
}

// Kernel 1, neighbored reduction in gpu. 
__global__  void reduceNeighbored(int *g_idata, int *g_odata, unsigned int n)
{
	// set thread id.
	unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

	// convert global data pointer to th local pointer of this block. 
	int *idata = g_idata + blockIdx.x * blockDim.x;

	// boundary check.
	if (idx >= n) return;

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

// Kernel 2, neighbored reduction in gpu with less divergence. 
__global__ void reduceNeighboredLess(int *g_idata, int *g_odata, unsigned int n)
{
    // set the thread id.
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // convert global data pointer to the local pointer of this block. 
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check.
    if (idx >= n) return;

    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        // convert tid into local array index. 
        int index = 2 * stride * tid;

        if (index < blockDim.x)
        {
            idata[index] += idata[index + stride];
        }

        // synchronize within threadblock.
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
	int *tmp1 = (int *)malloc(bytes);
	int *tmp2 = (int *)malloc(bytes);

	// initialize the array. 
	for (int i =0; i<size; i++)
	{
		// mask off high 2 bytes to force max number to 255. 
		h_idata[i] = (int)(rand() & 0xFF);
	}

	memcpy(tmp1, h_idata, bytes);
	memcpy(tmp2, h_idata, bytes);

	double iStart, iElaps;
	int gpu_sum = 0;

	// allocate device memory. 
	int *d_idata = NULL;
	int *d_odata = NULL;

	cudaMalloc((void **) &d_idata, bytes);
	cudaMalloc((void **) &d_odata, grid.x * sizeof(int));

	// cpu normal reduction.
	iStart = seconds();
	int cpu_sum = normalReduce(tmp1, size);
	iElaps = seconds() - iStart;
	printf("cpu normal reduce elapsed     %lf ms  seconds, sum value: %d\n", iElaps*1.e3, cpu_sum);
	
    // cpu recursive reduction. 
    iStart = seconds();
    cpu_sum = recursiveReduce(tmp2, size);
    iElaps = seconds() - iStart; 
	printf("cpu recursive reduce elapsed  %lf ms  seconds, sum value: %d\n", iElaps*1.e3, cpu_sum);

    // kernel 1: reduceNeighbored.
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    
    iStart = seconds();
    reduceNeighbored <<<grid, block>>> (d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;

    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    
    gpu_sum = 0;
    for (int i=0; i<grid.x; i++)
    {
        gpu_sum += h_odata[i];
    }
	printf("gpu neighbord reduce elapsed  %lf ms  seconds, sum value: %d\n", iElaps*1.e3, gpu_sum);
    
    // check the result. 
    if (gpu_sum != cpu_sum)
    {
        bResult = true;
    }

    // kernel 2: reduceNeighboredLess.
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    
    iStart = seconds();
    reduceNeighboredLess <<<grid, block>>> (d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;

    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    
    gpu_sum = 0;
    for (int i=0; i<grid.x; i++)
    {
        gpu_sum += h_odata[i];
    }
	printf("gpu neighbord reduceL elapsed %lf ms  seconds, sum value: %d\n", iElaps*1.e3, gpu_sum);
    
    // check the result. 
    if (gpu_sum != cpu_sum)
    {
        bResult = true;
    }


    if (bResult)
    {
        printf("Test failed!\n");
    }
    else
    {
        printf("Test succeed!\n");
    }

    // free host memory.
    free(h_idata);
    free(h_odata);

    // free device memory.
    CHECK(cudaFree(d_idata));
    CHECK(cudaFree(d_odata));

    // reset device. 
    CHECK(cudaDeviceReset());

	return EXIT_SUCCESS;
}







