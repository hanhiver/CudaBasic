#include <stdlib.h>
#include <stdio.h>
#include <time.h> 

typedef struct 
{
    int width; 
    int height; 
    float* elements; 
} Matrix; 

#define BLOCK_SIZE 2
#define MATRIX_SIZE 2

__global__ void MatMulKernel(const Matrix, const Matrix, const Matrix);

void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    Matrix d_A; 
    d_A.width = A.width; 
    d_A.height = A.height; 
    size_t size_a = A.width * A.height * sizeof(float); 
    cudaMalloc(&d_A.elements, size_a);
    cudaMemcpy(d_A.elements, A.elements, size_a, cudaMemcpyHostToDevice); 

    Matrix d_B; 
    d_B.width = B.width; 
    d_B.height = B.height; 
    size_t size_b = B.width * B.height * sizeof(float); 
    cudaMalloc(&d_B.elements, size_b);
    cudaMemcpy(d_B.elements, B.elements, size_b, cudaMemcpyHostToDevice); 

    Matrix d_C;
    d_C.width = C.width; 
    d_C.height = C.height;
    size_t size_c = B.width * B.height * sizeof(float); 
    cudaMalloc(&d_C.elements, size_c);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    
    cudaMemcpy(C.elements, d_C.elements, size_c, cudaMemcpyDeviceToHost);
    printf("MutMul: %f, %f", C.elements[0], C.elements[MATRIX_SIZE*MATRIX_SIZE-1]);
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    float Cvalue = 0; 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    for (int e=0; e<A.width; ++e)
    {
        Cvalue += A.elements[row*A.width+e] * B.elements[e*B.width+col]; 
    }
    C.elements[row*C.width+col] = Cvalue; 
    printf("Kernel: Cvalue = %f\n", Cvalue);
}

void initialData(float *ip, int size)
{
    time_t t; 
    srand((unsigned int) time(&t));

    for (int i=0; i<size; ++i)
    {
        ip[i] = (float)( rand() & 0xFF )/10.0f;
    }
}

int main()
{
    Matrix A; 
    A.width = MATRIX_SIZE; 
    A.height = MATRIX_SIZE;
    A.elements = (float*)malloc(sizeof(float)*MATRIX_SIZE*MATRIX_SIZE);
    initialData(A.elements, MATRIX_SIZE*MATRIX_SIZE);

    Matrix B; 
    B.width = MATRIX_SIZE; 
    B.height = MATRIX_SIZE;
    B.elements = (float*)malloc(sizeof(float)*MATRIX_SIZE*MATRIX_SIZE);
    initialData(A.elements, MATRIX_SIZE*MATRIX_SIZE);

    Matrix C; 
    C.width = MATRIX_SIZE; 
    C.height = MATRIX_SIZE;
    C.elements = (float*)malloc(sizeof(float)*MATRIX_SIZE*MATRIX_SIZE);

    printf("Main1, first element: %f, last element: %f\n", A.elements[0], A.elements[MATRIX_SIZE*MATRIX_SIZE-1]);
    
    MatMul(A, B, C); 

    printf("Main2, first element: %f, last element: %f\n", C.elements[0], C.elements[MATRIX_SIZE*MATRIX_SIZE-1]);

    free(A.elements);
    free(B.elements);
}