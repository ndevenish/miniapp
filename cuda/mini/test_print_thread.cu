#include <cstdio>

__global__ void PrintTest() {
    printf("Thread %3d,  %d,%d,%d\n", threadIdx.x, blockIdx.x, blockIdx.y, blockIdx.z);
}

int main() {
    dim3 blocks(10, 3, 1);
    PrintTest<<<blocks, 60>>>();
    cudaDeviceSynchronize();
}
