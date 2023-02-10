#include <cstdio>

__global__ void test_shuffle_xor() {
    int value = threadIdx.x;
    for (int i = 1; i < 32; i *= 2) {
        value += __shfl_xor_sync(-1, value, i);
    }
    printf("Thread %2d = %d\n", threadIdx.x, value);
}

__global__ void test_shuffle_down() {
    int value = threadIdx.x;
    for (int i = 16; i > 0; i = i / 2) {
        value += __shfl_down_sync(-1, value, i);
    }
    printf("Thread %2d = %d\n", threadIdx.x, value);
}
int main() {
    printf("XOR\n");
    test_shuffle_xor<<<1, 32>>>();
    cudaDeviceSynchronize();
    printf("Down\n");
    test_shuffle_down<<<1, 32>>>();
    cudaDeviceSynchronize();
}
