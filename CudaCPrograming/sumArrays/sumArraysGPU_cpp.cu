#include <iostream> 
#include <random> 
#include <algorithm> 
#include <chrono> 

void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
    int idx;
    for (idx=0; idx<N; ++idx)
    {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx<N)
    {
        C[idx] = A[idx] + B[idx];
    }
}

void initialData(float *ip, int size)
{
    std::mt19937 gen; 
    std::uniform_int_distribution<int> dis(0, 100000);

    auto rand_num ([=]() mutable
        {
            return dis(gen)/100.0f;
        });
    
    std::generate(ip, ip+size, rand_num); 
}

int main(int argc, char* argv[])
{
    int nElem = 1<<24; 
    size_t nBytes = nElem * sizeof(float);

    float *h_A, *h_B, *h_C; 
    h_A = new float[nElem]; 
    h_B = new float[nElem]; 
    h_C = new float[nElem]; 

    float *d_A, *d_B, *d_C; 
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_C, nBytes);

    initialData(h_A, nElem);
    initialData(h_B, nElem);

    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
    
    {
        auto start = std::chrono::steady_clock::now(); 
        sumArraysOnHost(h_A, h_B, h_C, nElem);
        auto end = std::chrono::steady_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end-start);
        double period = double(dur.count());

        std::cout << "CPU sum in " << period << " us. " << std::endl; 
    }

    {
        int iLen = 512; 
        dim3 block(iLen);
        dim3 grid((nElem+block.x-1)/block.x);

        auto start = std::chrono::steady_clock::now(); 
        sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
        cudaDeviceSynchronize(); 
        auto end = std::chrono::steady_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end-start);
        double period = double(dur.count());

        std::cout << "GPU sum in " << period << " us. " << std::endl; 
    }

    cudaDeviceReset();

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
}