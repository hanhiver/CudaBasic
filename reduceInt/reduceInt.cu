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

// kernel 3, interleaved pair implementation with less divergence.
__global__ void reduceInterleaved(int *g_idata, int *g_odata, unsigned int n)
{
	// set the thread id.
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // convert global data pointer to the local pointer of this block. 
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check.
    if (idx >= n) return;
	
	// in-place reduction in global memory
    for (int stride = blockDim.x/2; stride>0; stride>>=1)
    {
		if (tid < stride)
		{
			idata[tid] += idata[tid + stride];
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

// kernel 4, reduce unrolling 2.
__global__ void reduceUnrolling2(int *g_idata, int *g_odata, unsigned int n)
{
	// set the thread id.
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x * 2;

    // convert global data pointer to the local pointer of this block. 
    int *idata = g_idata + blockIdx.x * blockDim.x * 2;
	// unrolling 2 data blocks. 
	if (idx + blockDim.x < n)
	{
		g_idata[idx] += g_idata[idx + blockDim.x];
	}
	__syncthreads();
	
    // boundary check.
    if (idx >= n) return;
	
	// in-place reduction in global memory
    for (int stride = blockDim.x/2; stride>0; stride>>=1)
    {
		if (tid < stride)
		{
			idata[tid] += idata[tid + stride];
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

// kernel 5, reduce unrolling 8 with Warps8.
__global__ void reduceUnrollWarp8(int *g_idata, int *g_odata, unsigned int n)
{
	// set the thread id.
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x * 8;

    // convert global data pointer to the local pointer of this block. 
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

	// unrolling 8 data blocks. 
	if (idx + blockDim.x * 7 < n)
	{
        int a1 = g_idata[idx];
		int a2 = g_idata[idx + blockDim.x];
		int a3 = g_idata[idx + blockDim.x * 2];
		int a4 = g_idata[idx + blockDim.x * 3];
		int b1 = g_idata[idx + blockDim.x * 4];
		int b2 = g_idata[idx + blockDim.x * 5];
		int b3 = g_idata[idx + blockDim.x * 6];
		int b4 = g_idata[idx + blockDim.x * 7];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
	}
	__syncthreads();
	
	// in-place reduction in global memory
    for (int stride = blockDim.x/2; stride>32; stride>>=1)
    {
		if (tid < stride)
		{
			idata[tid] += idata[tid + stride];
		}

        // synchronize within threadblock.
        __syncthreads();
    }

    // unrolling warp
    if (tid < 32)
    {
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid +  8];
        vmem[tid] += vmem[tid +  4];
        vmem[tid] += vmem[tid +  2];
        vmem[tid] += vmem[tid +  1];
    }

    // write result for this block to global mem. 
    if (tid == 0) 
    {
        g_odata[blockIdx.x] = idata[0];
    }
}

// kernel 6, reduction complete unrolling 8 with Warps8.
__global__ void reduceCompUnrollW(int *g_idata, int *g_odata, unsigned int n)
{
	// set the thread id.
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x * 8;

    // convert global data pointer to the local pointer of this block. 
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

	// unrolling 8 data blocks. 
	if (idx + blockDim.x * 7 < n)
	{
        int a1 = g_idata[idx];
		int a2 = g_idata[idx + blockDim.x];
		int a3 = g_idata[idx + blockDim.x * 2];
		int a4 = g_idata[idx + blockDim.x * 3];
		int b1 = g_idata[idx + blockDim.x * 4];
		int b2 = g_idata[idx + blockDim.x * 5];
		int b3 = g_idata[idx + blockDim.x * 6];
		int b4 = g_idata[idx + blockDim.x * 7];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
	}
	__syncthreads();
	
    // in-place reduction and complete unroll
    if (blockDim.x >= 1024 && tid < 512) idata[tid] += idata[tid + 512];
    __syncthreads();
    if (blockDim.x >= 512 && tid < 256) idata[tid] += idata[tid + 256];
    __syncthreads();
    if (blockDim.x >= 256 && tid < 128) idata[tid] += idata[tid + 128];
    __syncthreads();
    if (blockDim.x >= 128 && tid < 64) idata[tid] += idata[tid + 64];
    __syncthreads();

    // unrolling warp
    if (tid < 32)
    {
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid +  8];
        vmem[tid] += vmem[tid +  4];
        vmem[tid] += vmem[tid +  2];
        vmem[tid] += vmem[tid +  1];
    }

    // write result for this block to global mem. 
    if (tid == 0) 
    {
        g_odata[blockIdx.x] = idata[0];
    }
}

// kernel 7, reduction complete unrolling 8 with Warps8 and template.
template <unsigned int iBlockSize>
__global__ void reduceCompUnroll(int *g_idata, int *g_odata, unsigned int n)
{
	// set the thread id.
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x * 8;

    // convert global data pointer to the local pointer of this block. 
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

	// unrolling 8 data blocks. 
	if (idx + blockDim.x * 7 < n)
	{
        int a1 = g_idata[idx];
		int a2 = g_idata[idx + blockDim.x];
		int a3 = g_idata[idx + blockDim.x * 2];
		int a4 = g_idata[idx + blockDim.x * 3];
		int b1 = g_idata[idx + blockDim.x * 4];
		int b2 = g_idata[idx + blockDim.x * 5];
		int b3 = g_idata[idx + blockDim.x * 6];
		int b4 = g_idata[idx + blockDim.x * 7];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
	}
	__syncthreads();
	
    // in-place reduction and complete unroll
    if (iBlockSize >= 1024 && tid < 512) idata[tid] += idata[tid + 512];
    __syncthreads();
    if (iBlockSize >= 512 && tid < 256) idata[tid] += idata[tid + 256];
    __syncthreads();
    if (iBlockSize >= 256 && tid < 128) idata[tid] += idata[tid + 128];
    __syncthreads();
    if (iBlockSize >= 128 && tid < 64) idata[tid] += idata[tid + 64];
    __syncthreads();

    // unrolling warp
    if (tid < 32)
    {
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid +  8];
        vmem[tid] += vmem[tid +  4];
        vmem[tid] += vmem[tid +  2];
        vmem[tid] += vmem[tid +  1];
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
	printf("cpu normal reduce elapsed         \t %lf ms \t seconds, sum value: %d\n", iElaps*1.e3, cpu_sum);
	
    // cpu recursive reduction. 
    iStart = seconds();
    cpu_sum = recursiveReduce(tmp2, size);
    iElaps = seconds() - iStart; 
	printf("cpu recursive reduce elapsed      \t %lf ms \t seconds, sum value: %d\n", iElaps*1.e3, cpu_sum);

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
	printf("gpu neighbord reduce elapsed      \t %lf ms \t seconds, sum value: %d\n", iElaps*1.e3, gpu_sum);
    
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
	printf("gpu neighbord reduceL elapsed     \t %lf ms \t seconds, sum value: %d\n", iElaps*1.e3, gpu_sum);
    
    // check the result. 
    if (gpu_sum != cpu_sum)
    {
        bResult = true;
    }

    // kernel 3: reduceInterleaved.
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    
    iStart = seconds();
    reduceInterleaved <<<grid, block>>> (d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;

    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    
    gpu_sum = 0;
    for (int i=0; i<grid.x; i++)
    {
        gpu_sum += h_odata[i];
    }
	printf("gpu interleaved reduceL elapsed   \t %lf ms \t seconds, sum value: %d\n", iElaps*1.e3, gpu_sum);
    
    // check the result. 
    if (gpu_sum != cpu_sum)
    {
        bResult = true;
    }

    // kernel 4: reduceUnrolling2.
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    
    iStart = seconds();
	
	// after rolling 2 block, we only need half of the grid. 
    reduceUnrolling2 <<<grid.x / 2, block>>> (d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;

    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    
    gpu_sum = 0;
    for (int i=0; i<grid.x/2; i++)
    {
        gpu_sum += h_odata[i];
    }
	printf("gpu unrolling2 reduceL elapsed    \t %lf ms \t seconds, sum value: %d\n", iElaps*1.e3, gpu_sum);
    
    // check the result. 
    if (gpu_sum != cpu_sum)
    {
        bResult = true;
    }

    // kernel 5: reduceUnrollWarp8.
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    
    iStart = seconds();
	
	// after rolling 2 block, we only need half of the grid. 
    reduceUnrollWarp8 <<<grid.x / 8, block>>> (d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;

    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int), cudaMemcpyDeviceToHost));
    
    gpu_sum = 0;
    for (int i=0; i<grid.x/8; i++)
    {
        gpu_sum += h_odata[i];
    }
	printf("gpu unrollWarp8 reduceL elapsed   \t %lf ms \t seconds, sum value: %d\n", iElaps*1.e3, gpu_sum);
    
    // check the result. 
    if (gpu_sum != cpu_sum)
    {
        bResult = true;
    }

    // kernel 6: reduceCompUnroll.
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    
    iStart = seconds();
	
	// after rolling 8 block, we only need 1/8 of the grid. 
    reduceCompUnrollW <<<grid.x / 8, block>>> (d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;

    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int), cudaMemcpyDeviceToHost));
    
    gpu_sum = 0;
    for (int i=0; i<grid.x/8; i++)
    {
        gpu_sum += h_odata[i];
    }
	printf("gpu CompunrollW reduceL elapsed   \t %lf ms \t seconds, sum value: %d\n", iElaps*1.e3, gpu_sum);
    
    // check the result. 
    if (gpu_sum != cpu_sum)
    {
        bResult = true;
    }

    // kernel 7: reduceCompUnroll with template.
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    
    iStart = seconds();
	
    // Use switch to eliminate the non-access code during compiling. 
    switch (blocksize)
    {
    case 1024:
        reduceCompUnroll<1024><<<grid.x / 8, block>>>(d_idata, d_odata, size);
        break;

    case 512:
        reduceCompUnroll<512><<<grid.x / 8, block>>>(d_idata, d_odata, size);
        break;

    case 256:
        reduceCompUnroll<256><<<grid.x / 8, block>>>(d_idata, d_odata, size);
        break;

    case 128:
        reduceCompUnroll<128><<<grid.x / 8, block>>>(d_idata, d_odata, size);
        break;

    case 64:
        reduceCompUnroll<64><<<grid.x / 8, block>>>(d_idata, d_odata, size);
        break;
    }

	// after rolling 8 block, we only need 1/8 of the grid. 
    //reduceCompUnrollW <<<grid.x / 8, block>>> (d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;

    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int), cudaMemcpyDeviceToHost));
    
    gpu_sum = 0;
    for (int i=0; i<grid.x/8; i++)
    {
        gpu_sum += h_odata[i];
    }
	printf("gpu Compunroll reduceL elapsed    \t %lf ms \t seconds, sum value: %d\n", iElaps*1.e3, gpu_sum);
    
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







