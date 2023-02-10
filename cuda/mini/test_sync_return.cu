#include <cstdio>

__global__ void test_sync() {
    if (threadIdx.x == 2) return;
    __syncthreads();
    printf("Post-sync thread: %2d\n", threadIdx.x);
}

int main() {
    int a = 2;

    test_sync<<<dim3(a), 32>>>();
    cudaDeviceSynchronize();
    printf("Fin.\n");
}
