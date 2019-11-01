#include <iostream> 
#include <cstdio> 

__global__ void helloFromGPU(void)
{
    printf("Hello from GPU - block: %d - thread: %d. \n", blockIdx.x, threadIdx.x);
}

int main()
{
    std::cout << "Hello from CPU. " << std::endl; 
    helloFromGPU<<<2, 5>>>(); 
    //cudaDeviceReset();
    cudaDeviceSynchronize();
    return 0; 
}