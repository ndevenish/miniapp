/**
 * Basic Summing with CUDA
 * 
 * Uses h5read to loop over all images, calculates a pixel sum in host
 * and GPU, and compares the results.
 * 
 * Demonstrates using h5read and GPU reduction.
 * 
 */

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <fmt/core.h>

#include <array>
#include <cassert>
#include <memory>

#include "common.hpp"
#include "h5read.h"

namespace cg = cooperative_groups;

using namespace fmt;

using pixel_t = H5Read::image_type;

/// GPU Kernel to sum a whole image
__global__ void do_sum_image(size_t *block_store,
                             pixel_t *data,
                             size_t pitch,
                             size_t width,
                             size_t height) {
    // Store an interim block sum int for every warp.
    // In theory this could be less than 32, because we might not have
    // launched the maximum threads. However, then we would need to
    // calculate and pass shared memory requirements on launch. On all
    // cards max_threads <= 1024 (32 warps), so settings to 32 is safe.
    static __shared__ size_t shared[32];

    // Use co-operative groups for measurements
    auto g = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(g);

    int warpId = warp.meta_group_rank();
    int lane = warp.thread_rank();

    // Grid-stride looping.
    //
    // This ensures that we always cover the entire image, even if the
    // combination of grid and block shapes wouldn't. This allows us to
    // optimise the shape of the grid seperately from the size of the
    // image, instead of being forced into 1 thread = 1 pixel
    //
    // See https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
    size_t sum = 0;
    for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < height;
         y += blockDim.y * gridDim.y) {
        for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < width;
             x += blockDim.x * gridDim.x) {
            sum += data[y * (pitch / sizeof(pixel_t)) + x];
        }
    }

    // Sum the current warp
    sum = cg::reduce(warp, (int)sum, cg::plus<int>());

    // And save the warp-total to shared memory
    if (lane == 0) {
        shared[warpId] = sum;
    }

    // Wait for all thread warps in the block to write
    g.sync();

    if (warpId == 0) {
        // Load the shared memory value for the warp corresponding to this lane.
        // We need to check this, because although we have a maximum number
        // of warps per block (32), we might have had less than that.
        sum = (lane < (blockDim.x * blockDim.y) / warpSize) ? shared[lane] : 0;
        // And sum all of the warps in this block together
        sum = cg::reduce(warp, sum, cg::plus<size_t>());
        // Finally, store the block total sum, once.
        if (lane == 0) {
            int blockId = blockIdx.x + gridDim.x * blockIdx.y;
            block_store[blockId] = sum;
        }
    }
}

int main(int argc, char **argv) {
    // Parse arguments and get our H5Reader
    auto parser = CUDAArgumentParser();
    parser.add_h5read_arguments();
    auto args = parser.parse_args(argc, argv);

    auto reader = args.file.empty() ? H5Read() : H5Read(args.file);

    int height = reader.image_shape()[0];
    int width = reader.image_shape()[1];

    // Work out how many blocks this is
    dim3 thread_block_size{32, 16};
    dim3 blocks_dims{
      static_cast<unsigned int>(ceilf((float)width / thread_block_size.x)),
      static_cast<unsigned int>(ceilf((float)height / thread_block_size.y))};
    const int num_threads_per_block = thread_block_size.x * thread_block_size.y;
    const int num_blocks = blocks_dims.x * blocks_dims.y;
    print("Image:   {:4d} x {:4d} = {} px\n", width, height, width * height);
    print("Threads: {:4d} x {:<4d} = {}\n",
          thread_block_size.x,
          thread_block_size.y,
          num_threads_per_block);
    print("Blocks:  {:4d} x {:<4d} = {}\n", blocks_dims.x, blocks_dims.y, num_blocks);

    // Create a host memory area to store the current image
    auto host_image = std::make_unique<pixel_t[]>(width * height);

    // Create a device-side pitched area
    pixel_t *dev_image = nullptr;
    size_t device_pitch = 0;
    cudaMallocPitch(&dev_image, &device_pitch, width * sizeof(pixel_t), height);
    print("Allocated device memory. Pitch = {} vs naive {}\n",
          device_pitch,
          width * sizeof(pixel_t));
    cuda_throw_error();

    // And a device-side location to store per-block results
    size_t *dev_result = nullptr;
    cudaMalloc(&dev_result, sizeof(decltype(*dev_result)) * num_blocks);
    cuda_throw_error();

    for (size_t image_id = 0; image_id < reader.get_number_of_images(); ++image_id) {
        print("Image {}:\n", image_id);
        reader.get_image_into(image_id, host_image.get());

        // Calculate the sum of all pixels host-side
        size_t sum = 0;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                sum += host_image[x + y * width];
            }
        }
        print("    Summed pixels: {}\n", bold(sum));

        cudaEvent_t start, mid_pre, mid_post, end;
        cudaEventCreate(&start);
        cudaEventCreate(&mid_pre);
        cudaEventCreate(&mid_post);
        cudaEventCreate(&end);

        cudaEventRecord(start, 0);
        // Copy the image to GPU
        cudaMemcpy2D(dev_image,
                     device_pitch,
                     host_image.get(),
                     width * sizeof(pixel_t),
                     width * sizeof(pixel_t),
                     height,
                     cudaMemcpyHostToDevice);
        cuda_throw_error();
        cudaEventRecord(mid_pre, 0);

        // Launch the kernel to sum each block
        do_sum_image<<<blocks_dims, thread_block_size>>>(
          dev_result, dev_image, device_pitch, width, height);
        cudaEventRecord(mid_post, 0);
        cudaDeviceSynchronize();
        cuda_throw_error();

        // Copy the per-block sum data back, to sum (CPU-side for now)
        auto host_result =
          std::make_unique<std::remove_reference<decltype(*dev_result)>::type[]>(
            num_blocks);
        cudaMemcpy(host_result.get(),
                   dev_result,
                   sizeof(decltype(*dev_result)) * num_blocks,
                   cudaMemcpyDeviceToHost);
        cuda_throw_error();
        cudaEventRecord(end, 0);

        // Manually sum the response here
        size_t accum = 0;
        for (int i = 0; i < num_blocks; ++i) {
            accum += host_result[i];
        }
        if (accum == sum) {
            print("    Kernel Summed: {}\n", green(bold(accum)));
        } else {
            print("    Kernel Summed: {}\n", red(bold(accum)));
        }

        // Print timings
        float elapsedTime_pre, elapsedTime_post, elapsedTime_end;
        cudaEventElapsedTime(&elapsedTime_pre, start, mid_pre);
        cudaEventElapsedTime(&elapsedTime_post, start, mid_post);
        cudaEventElapsedTime(&elapsedTime_end, start, end);
        print("        Time Memcpy: {:6.2f} ms\n", elapsedTime_pre);
        print(" Time Memcpy+Kernel: {:6.2f} ms\n", elapsedTime_post);
        print("           Time All: {:6.2f} ms\n", elapsedTime_end);
        print("\n");

        cudaEventDestroy(start);
        cudaEventDestroy(mid_pre);
        cudaEventDestroy(mid_post);
        cudaEventDestroy(end);
    }
    cudaFree(dev_result);
    cudaFree(dev_image);
}
