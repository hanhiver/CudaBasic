#include <cuda_runtime.h>
#include <stdio.h>

#include "../common.h"

#define LEN 1<<24

struct innerStruct
{
	float x; 
	float y;
};

struct innerArray
{
	float x[LEN];
	float y[LEN];
};

void initialInnerStruct(innerStruct *ip, int size)
{
	for (int i=0; i<size; i++)
	{
		ip[i].x = (float)(rand() & 0xFF) / 100.0f;
		ip[i].y = (float)(rand() & 0xFF) / 100.0f;
	}

	return;
}

void initialInnerArray(innerArray *ip, int size)
{
	for (int i=0; i<size; i++)
	{
		ip->x[i] = (float)(rand() & 0xFF) / 100.0f;
		ip->y[i] = (float)(rand() & 0xFF) / 100.0f;
	}

	return;
}

void testInnerStructHost(innerStruct *A, innerStruct *C, const int n)
{
	for (int idx=0; idx<n; idx++)
	{
		C[idx].x = A[idx].x + 10.f;
		C[idx].y = A[idx].y + 20.f;
	}
	
	return;
}

void testInnerArrayHost(innerArray *A, innerArray *C, const int size)
{
	for (int idx=0; idx<size; idx++)
	{
		C->x[idx] = A->x[idx] + 10.f;
		C->y[idx] = A->y[idx] + 20.f;
	}

	return;
}

void checkInnerArray(innerArray *hostRef, innerArray *gpuRef, const int size)
{
	double epsilon = 1.0E-8;
	bool match = 1; 

	for (int i=0; i<size; i++)
	{
		if (abs(hostRef->x[i] - gpuRef->x[i]) > epsilon)
		{
			match = 0;
			printf("Different on %dth element: host %f gpu %f\n", 
					i, hostRef->x[i], gpuRef->x[i]);
			break;
		}
		
		if (abs(hostRef->y[i] - gpuRef->y[i]) > epsilon)
		{
			match = 0;
			printf("Different on %dth element: host %f gpu %f\n", 
					i, hostRef->y[i], gpuRef->y[i]);
			break;
		}
	}

	if (!match)
		printf("Arrays do not match!. \n\n");
}


void checkInnerStruct(innerStruct *hostRef, innerStruct *gpuRef, const int n)
{
	double epsilon = 1.0E-8;
	bool match = 1;

	for (int i=0; i<n; i++)
	{
		if (abs(hostRef[i].x - gpuRef[i].x) > epsilon)
		{
			match = 0; 
			printf("Different on %dth element: host %f gpu %f\n", i, 
					hostRef[i].x, gpuRef[i].x);
			break;
		}

		if (abs(hostRef[i].y - gpuRef[i].y) > epsilon)
		{
			match = 0; 
			printf("Different on %dth element: host %f gpu %f\n", i, 
					hostRef[i].y, gpuRef[i].y);
			break;
		}
	}

	if (!match)
		printf("Arrays do not match!. \n\n");
}

__global__ void testInnerArray(innerArray *data, innerArray *result, const int size)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i<size)
	{
		result->x[i] = data->x[i] + 10.f;
		result->y[i] = data->y[i] + 20.f;
	}
}

__global__ void warmup2(innerArray *data, innerArray *result, const int size)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i<size)
	{
		result->x[i] = data->x[i] + 10.f;
		result->y[i] = data->y[i] + 20.f;
	}
}

__global__ void testInnerStruct(innerStruct *data, innerStruct *result, const int n)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i<n)
	{
		innerStruct tmp = data[i];
		tmp.x += 10.f;
		tmp.y += 20.f;
		result[i] = tmp;
	}
}

__global__ void warmup1(innerStruct *data, innerStruct *result, const int n)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i<n)
	{
		innerStruct tmp = data[i];
		tmp.x += 10.f;
		tmp.y += 20.f;
		result[i] = tmp;
	}
}

int main(int argc, char ** argv)
{
	// set up device. 
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("%s test struct of array at ", argv[0]);
	printf("device %d: %s \n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));

	// allocate host memory. 
	int nElem = LEN;
	size_t nBytes = nElem * sizeof(innerStruct);
	innerStruct     *h_A = (innerStruct *)malloc(nBytes);
	innerStruct *hostRef = (innerStruct *)malloc(nBytes);
	innerStruct  *gpuRef = (innerStruct *)malloc(nBytes);

	// initialize host array. 
	initialInnerStruct(h_A, nElem);
	testInnerStructHost(h_A, hostRef, nElem);

	// allocate device memory. 
	innerStruct *d_A, *d_C;
	CHECK(cudaMalloc((innerStruct **) &d_A, nBytes));
	CHECK(cudaMalloc((innerStruct **) &d_C, nBytes));

	// copy data from host to device. 
	CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));

	// set up blocksize for summaryAU.	
	int blocksize = 128;

	if (argc > 1)
		blocksize = atoi(argv[1]);

	// execution configuration. 
	dim3 block (blocksize, 1);
	dim3 grid ((nElem + block.x - 1) / block.x, 1);

    printf("===== Structure of Array (SOA) test =====\n");

	// kernel 1: warmup.
	double iStart = seconds();
	warmup1 <<<grid, block>>> (d_A, d_C, nElem);
	CHECK(cudaDeviceSynchronize());
	double iElaps = seconds() - iStart;
	printf("warmup1     <<< %3d, %3d >>> elapsed %f sec.\n", 
			grid.x, block.x, iElaps);

	CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
	checkInnerStruct(hostRef, gpuRef, nElem);
	CHECK(cudaGetLastError());

	// kernel 2 testInnerStruct.
	iStart = seconds();
	testInnerStruct <<<grid, block>>> (d_A, d_C, nElem);
	CHECK(cudaDeviceSynchronize());
	iElaps = seconds() - iStart;
	printf("innerstruct <<< %3d, %3d >>> elapsed %f sec.\n", 
			grid.x, block.x, iElaps);
	
    printf("===== Array of Structure (AOS) test =====\n");
	
    // allocate host memory. 
    innerArray *h_Aa     = (innerArray *)malloc(nBytes);
	innerArray *hostRefa = (innerArray *)malloc(nBytes);
	innerArray *gpuRefa  = (innerArray *)malloc(nBytes);

	// initialize host array. 
	initialInnerArray(h_Aa, nElem);
	testInnerArrayHost(h_Aa, hostRefa, nElem);

	// allocate device memory. 
	innerArray *d_Aa, *d_Ca;
	CHECK(cudaMalloc((innerArray **) &d_Aa, nBytes));
	CHECK(cudaMalloc((innerArray **) &d_Ca, nBytes));

	// copy data from host to device. 
	CHECK(cudaMemcpy(d_Aa, h_Aa, nBytes, cudaMemcpyHostToDevice));

    printf("===== Array of Structure (AOS) test =====\n");
    
    // kernel 3 warmup2.
	iStart = seconds();
	warmup2 <<<grid, block>>> (d_Aa, d_Ca, nElem);
	CHECK(cudaDeviceSynchronize());
	iElaps = seconds() - iStart;
	printf("warmup2     <<< %3d, %3d >>> elapsed %f sec.\n", 
			grid.x, block.x, iElaps);

    // kernel 3 testInnerArray.
	iStart = seconds();
	testInnerArray <<<grid, block>>> (d_Aa, d_Ca, nElem);
	CHECK(cudaDeviceSynchronize());
	iElaps = seconds() - iStart;
	printf("innerArray  <<< %3d, %3d >>> elapsed %f sec.\n", 
			grid.x, block.x, iElaps);

    // free memories both host and device. 
	CHECK(cudaFree(d_A));
	CHECK(cudaFree(d_C));
	free(h_A);
	free(hostRef);
	free(gpuRef);
    
    CHECK(cudaFree(d_Aa));
	CHECK(cudaFree(d_Ca));
	free(h_Aa);
	free(hostRefa);
	free(gpuRefa);

	// reset device.
	CHECK(cudaDeviceReset());
	return EXIT_SUCCESS;
}



