#include <iostream> 
#include <random> 
#include <algorithm> 

void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
    int idx;
    for (idx=0; idx<N; ++idx)
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
    int nElem = 10; 
    size_t nBytes = nElem * sizeof(float);

    float *h_A, *h_B, *h_C; 
    h_A = new float[nElem]; 
    h_B = new float[nElem]; 
    h_C = new float[nElem]; 

    initialData(h_A, nElem);
    initialData(h_B, nElem);

    sumArraysOnHost(h_A, h_B, h_C, nElem);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
}