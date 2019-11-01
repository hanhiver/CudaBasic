#include <cuda_runtime.h>
#include <stdio.h>

#include <sys/time.h>

#define CHECK(call)\
{\
	const cudaError_t error = call;\
	if (error != cudaSuccess)\
	{\
		printf("Error: %s:%d, ", __FILE__, __LINE__);\
		printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));\
		exit(1);\
	}\
}

double cpuSecond()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec*1.0e-6);
}

void checkResult(float *hostRef, float *gpuRef, const int N)
{
	double epsilon = 1.0E-8;
	bool match = 1;

	for (int i=0; i<N; i++)
	{
		if ( abs(hostRef[i] - gpuRef[i]) > epsilon )
		{
			match = 0;
			printf("Arrays do not match! \n");
			printf("host: %5.2f, gpu: %5.2f at current %d \n", hostRef[i], gpuRef[i], i);
			break;
		}
	}

	if (match)
		printf("Array match.\n\n");
}

void initialData(float *ip, int size)
{
	// Generate different seed for random number.
	time_t t;
	srand((unsigned int)time(&t));

	for (int i=0; i<size; i++)
	{
		ip[i] = (float)(rand() & 0xFF )/10.0f;
	}
}

void sumArrayOnHost(float *A, float *B, float *C, const int N)
{
	for (int idx=0; idx<N; idx++)
	{
		C[idx] = A[idx] + B[idx];
	}
}

__global__ void sumArraysOnGPU(float *A, float *B, float *C)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	C[i] = A[i] + B[i];
}

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny)
{
    float *ia = A;
    float *ib = B;
    float *ic = C;

    for (int iy=0; iy<ny; iy++)
    {
        for (int ix=0; ix<nx; ix++)
        {
            ic[ix] = ia[ix] + ib[ix];
            //printf("CPU Add: %f + %f = %f.\n", ia[ix], ib[ix], ic[ix]);
        }
        
        ia += nx;
        ib += nx;
        ic += nx;
    }
}

__global__ void sumMatrixOnGPU(float *MatA, float *MatB, float *MatC, int nx, int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    //printf("nx: %d, ny: %d, ix: %d, iy: %d, idx: %d\n", nx, ny, ix, iy, idx);

    if (ix<nx && iy<ny)
    {
        MatC[idx] = MatA[idx] + MatB[idx];
        //printf("GPU Add: %f + %f = %f.\n", MatA[idx], MatB[idx], MatC[idx]);
    }
}


int main(int argc, char **argv)
{
	printf("%s Strarting...\n", argv[0]);

	// set up device
	int dev = 0;
	cudaSetDevice(dev);

	// set up data size of vectors
	int nx = 1<<14; 
    int ny = 1<<14;
    int nxy = nx * ny;
    size_t nBytes = nxy * sizeof(float);
    printf("Vector size %d\n", nxy);
	
	// malloc host memory
	float *h_A, *h_B, *hostRef, *gpuRef;
	h_A = (float *)malloc(nBytes);
	h_B = (float *)malloc(nBytes);

	hostRef = (float *)malloc(nBytes);
	gpuRef = (float *)malloc(nBytes);
	
	// initialize data at host side
	initialData(h_A, nxy);
	initialData(h_B, nxy);

	memset(hostRef, 0, nBytes);
	memset(gpuRef, 0, nBytes);

	// malloc gpu global memory
	float *d_A, *d_B, *d_C;
	cudaMalloc((float **)&d_A, nBytes);
	cudaMalloc((float **)&d_B, nBytes);
	cudaMalloc((float **)&d_C, nBytes);

	// transfer data from host to gpu
	cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
	
	// invoke kernel at host side
	int dimx = 32;
    int dimy = 32;

    if (argc > 2)
    {
        dimx = atoi(argv[1]);
        dimy = atoi(argv[2]);
        printf("Customized dimx: %d, dimy %d.\n", dimx, dimy);
    }

    dim3 block (dimx, dimy);
	dim3 grid ( (nx + block.x - 1)/block.x, (ny + block.y -1)/block.y );
    
    // start time
	double time_gpu_start = cpuSecond();

	sumMatrixOnGPU<<<grid, block>>>(d_A, d_B, d_C, nx, ny);
    cudaDeviceSynchronize();
    
    // gpu finished time
	double time_gpu_finish = cpuSecond();
   
    // copy kernel result back to host side
	cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

	printf("Execution configuration <<<(%d, %d), (%d, %d)>>>\n", 
            grid.x, grid.y, block.x, block.y);
    
    // reset start time for CPU. 
    double time_cpu_start = cpuSecond();

	// add vector at host side for result check. 
	sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);

	// cpu finished time
	double time_cpu_finish = cpuSecond();

	// Check device results
	checkResult(hostRef, gpuRef, nxy);

	// free device global memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	
	// free host memory
	free(h_A);
	free(h_B);
	free(hostRef);
	free(gpuRef);
    
    double cpu_time = time_cpu_finish - time_cpu_start;
    double gpu_time = time_gpu_finish - time_gpu_start;

	printf("CPU job Done in %lf. \n", time_cpu_finish - time_cpu_start);
	printf("GPU job Done in %lf. \n", time_gpu_finish - time_gpu_start);
    printf("Accelarate ratio: %lf%%. \n", (cpu_time/gpu_time)*100.0);
	return(0);
}


