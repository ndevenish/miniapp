#include <fmt/core.h>

#include <array>
#include <memory>

#include "common.hpp"
#include "h5read.h"
using namespace fmt;

using pixel_t = H5Read::image_type;

__inline__ __device__ int warpReduceSum(int val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync((unsigned int)-1, val, offset);
    return val;
}

__global__ void do_sum_image(int *block_store,
                             pixel_t *data,
                             size_t pitch,
                             size_t width,
                             size_t height) {
    // Store an int for every warp. On all cards max_threads <= 1024 (32 warps)
    static __shared__ int shared[32];

    int warpId = (threadIdx.x + blockDim.x * threadIdx.y) / warpSize;
    int lane = (threadIdx.x + blockDim.x * threadIdx.y) % warpSize;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width) return;
    if (y >= height) return;

    pixel_t pixel = data[y * pitch + x];

    int sum = warpReduceSum(pixel);
    // Once per warp, store the sum of the whole block
    if (lane == 0) {
        shared[warpId] = sum;
    }
    __syncthreads();
    // Load each of the shared values into a single warp. This works
    // because maximum #warps <= warpSize
    sum = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
    if (lane == 0) {
        sum = warpReduceSum(sum);
    }
    if (warpId == 0 && lane == 0) {
        int blockId = blockIdx.x + gridDim.x * blockIdx.y;
        block_store[blockId] = sum;
    }
}

int main(int argc, char **argv) {
    // Parse arguments and get our H5Reader
    auto parser = CUDAArgumentParser();
    parser.add_h5read_arguments();
    auto args = parser.parse_args(argc, argv);

    auto reader = args.file.empty() ? H5Read() : H5Read(args.file);

    auto image_shape = reader.image_shape();
    int width = image_shape[1];
    int height = image_shape[0];

    // Work out how many blocks this is
    dim3 thread_block_size{32, 16};
    dim3 blocks_dims{static_cast<unsigned int>(ceilf(width / thread_block_size.x)),
                     static_cast<unsigned int>(ceilf(height / thread_block_size.y))};
    const int num_threads_per_block = thread_block_size.x * thread_block_size.y;
    const int num_blocks = blocks_dims.x * blocks_dims.y;
    print("Threads: {:4d} x {:<4d} = {}\n",
          thread_block_size.x,
          thread_block_size.y,
          num_threads_per_block);
    print("Blocks:  {:4d} x {:<4d} = {}\n", blocks_dims.x, blocks_dims.y, num_blocks);

    // Create a host memory area to store the current image
    auto host_image = std::make_unique<pixel_t[]>(image_shape[0] * image_shape[1]);

    // Create a device-side pitched area
    pixel_t *dev_image = nullptr;
    size_t device_pitch = 0;
    cudaMallocPitch(&dev_image,
                    &device_pitch,
                    image_shape[1] * sizeof(pixel_t),
                    image_shape[0] * sizeof(pixel_t));
    // And a device-side location to store results
    int *dev_result = nullptr;
    cudaMalloc(&dev_result, sizeof(int) * num_blocks);

    // Check that this all worked
    cuda_throw_error();
    print("Allocated device memory. Pitch = {} vs naive {}\n",
          device_pitch,
          image_shape[1] * sizeof(pixel_t));

    for (size_t image_id = 0; image_id < reader.get_number_of_images(); ++image_id) {
        print("Image {}:\n", image_id);
        reader.get_image_into(image_id, host_image.get());

        // Calculate the sum of all pixels host-side
        uint32_t sum = 0;
        for (int y = 0; y < image_shape[0]; ++y) {
            for (int x = 0; x < image_shape[1]; ++x) {
                sum += host_image[x + y * image_shape[1]];
            }
        }
        print("    Summed pixels: {}\n", bold(sum));

        // Copy to device
        cudaMemcpy2D(dev_image,
                     device_pitch,
                     dev_image,
                     image_shape[1] * sizeof(pixel_t),
                     image_shape[1] * sizeof(pixel_t),
                     image_shape[2] * sizeof(pixel_t),
                     cudaMemcpyHostToDevice);

        // Launch the summing kernel
        do_sum_image<<<blocks_dims, thread_block_size>>>(
          dev_result, dev_image, device_pitch, width, height);
        cudaDeviceSynchronize();

        auto host_result = std::make_unique<int[]>(num_blocks);
        cudaMemcpy(host_result.get(),
                   dev_result,
                   sizeof(int) * num_blocks,
                   cudaMemcpyDeviceToHost);

        // Manually sum the response here
        int accum = 0;
        for (int i = 0; i < num_blocks; ++i) {
            accum += host_result[i];
        }
        print("    Kernel Summed: {}\n", bold(accum));

        // int zeros = 0;
        // for (int i = 0; i < num_blocks; ++i) {
        //     int val = host_result[i];
        //     if (val == 0) {
        //         zeros += 1;
        //     } else {
        //         if (zeros == 1) {
        //             print("0  ");
        //             zeros = 0;
        //         } else if (zeros > 1) {
        //             print("... {} zeros ... ", zeros);
        //             zeros = 0;
        //         }
        //         print("{}  ", host_result[i]);
        //     }
        // }
        // if (zeros > 0) {
        //     print("... {} zeros. ", zeros);
        // }
        print("\n");
        // break;
    }
    cudaFree(dev_result);
    cudaFree(dev_image);
}
