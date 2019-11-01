#include <iostream> 
#include <cstdlib> 

__global__ void helloFromGPU(void)
{
    printf("Hello from GPU. \n");
}

int main()
{
    std::cout << "Hello from CPU. " << std::endl; 
    helloFromGPU<<<1, 10>>>(); 
    cudaDeviceReset();
    return 0; 
}