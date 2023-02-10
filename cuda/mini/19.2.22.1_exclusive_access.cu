__device__ __managed__ int x, y = 2;
__global__ void kernel() {
    x = 10;
}
int main() {
    kernel<<<1, 1>>>();
    y = 20;  // Error on GPUs not supporting concurrent access

    cudaDeviceSynchronize();
    return 0;
}
