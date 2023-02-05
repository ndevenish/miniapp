/**
 * Basic Summing with CUDA
 * 
 * Uses h5read to loop over all images, calculates a pixel sum in host
 * and GPU, and compares the results.
 * 
 * Demonstrates using h5read and GPU reduction.
 * 
 */

#include <fmt/core.h>

#include <array>
#include <cassert>
#include <memory>

#include "common.hpp"
#include "h5read.h"
using namespace fmt;

using pixel_t = H5Read::image_type;

template <typename T>
__inline__ __device__ auto warpReduceSum_sync(T val) -> T {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync((unsigned int)-1, val, offset);
    return val;
}

/// Draw a subset of the pixel values for a 2D image array
/// fast, slow, width, height - describe the bounding box to draw
/// data_width, data_height - describe the full data array size
template <typename T>
void draw_image_data(const T *data,
                     size_t fast,
                     size_t slow,
                     size_t width,
                     size_t height,
                     size_t data_width,
                     size_t data_height) {
    std::string format_type = "";
    if constexpr (std::is_integral<T>::value) {
        format_type = "d";
    } else {
        format_type = ".1f";
    }

    // Walk over the data and get various metadata for generation
    // Maximum value
    T accum = 0;
    // Maximum format width for each column
    std::vector<int> col_widths;
    for (int col = fast; col < fast + width; ++col) {
        size_t maxw = fmt::formatted_size("{}", col);
        for (int row = slow; row < min(slow + height, data_height); ++row) {
            auto val = data[col + data_width * row];
            auto fmt_spec = fmt::format("{{:{}}}", format_type);
            maxw = std::max(maxw, fmt::formatted_size(fmt_spec, val));
            accum = max(accum, val);
        }
        col_widths.push_back(maxw);
    }

    // Draw a row header
    fmt::print("x =       ");
    for (int i = 0; i < width; ++i) {
        auto x = i + fast;
        fmt::print("{:{}} ", x, col_widths[i]);
    }
    fmt::print("\n         ┌");
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < col_widths[i]; ++j) {
            fmt::print("─");
        }
        fmt::print("─");
    }
    fmt::print("┐\n");

    for (int y = slow; y < min(slow + height, data_height); ++y) {
        if (y == slow) {
            fmt::print("y = {:2d} │", y);
        } else {
            fmt::print("    {:2d} │", y);
        }
        for (int i = fast; i < fast + width; ++i) {
            // Calculate color
            // Black, 232->255, White
            // Range of 24 colors, not including white. Split into 25 bins, so
            // that we have a whole black top bin
            // float bin_scale = -25
            auto dat = data[i + data_width * y];
            int color = 255 - ((float)dat / (float)accum) * 24;
            if (color <= 231) color = 0;
            if (dat < 0) {
                color = 9;
            }

            if (dat == accum) {
                fmt::print("\033[0m\033[1m");
            } else {
                fmt::print("\033[38;5;{}m", color);
            }
            auto fmt_spec =
              fmt::format("{{:{}{}}} ", col_widths[i - fast], format_type);
            fmt::print(fmt_spec, dat);
            if (dat == accum) {
                fmt::print("\033[0m");
            }
        }
        fmt::print("\033[0m│\n");
    }
}

__global__ void fill(pixel_t *data, size_t size) {
    for (int i = 0; i < size; ++i) {
        data[i] = 3;
    }
}

/// Print out diagnostics of an image memory area
__global__ void diagnose_memory(pixel_t *data,
                                size_t pitch,
                                size_t width,
                                size_t height) {
    int sum = 0;
    // int calc_width = pitch / sizeof(pixel_t);
    int item_pitch = pitch / sizeof(pixel_t);
    for (int y = 0; y < height; ++y) {
        int last_val = -1;
        int count = 0;
        printf("%4d:  ", y);
        for (int x = 0; x < item_pitch; ++x) {
            if (x == width) {
                if (count > 0) {
                    printf("%d×%-4d ", last_val, count);
                    count = 0;
                }
                printf(" | ");
            }
            size_t index = y * item_pitch + x;
            pixel_t val = data[index];
            sum += val;
            if (val != last_val) {
                if (count > 0) {
                    printf("%d×%-4d ", last_val, count);
                }
                if (val == 3) {
                    printf(">%d< ", x);
                }
                last_val = val;
                count = 0;
            }
            count += 1;
        }
        if (count > 0) {
            printf("%d×%-4d ", last_val, count);
        }
        printf("\n");
    }
    printf("Total: %d\n", sum);
}

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

    int warpId = (threadIdx.x + blockDim.x * threadIdx.y) / warpSize;
    int lane = (threadIdx.x + blockDim.x * threadIdx.y) % warpSize;

    // Which position on the image do we need to read
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Always run the warp lane, even if out of image range. This is
    // because the warp shuffle functions will return the input number
    // if the target source is inactive, which gives us phantom numbers
    // for out-of-range pixels.
    size_t pixel = 0;
    if (x < width && y < height) {
        pixel = data[y * (pitch / sizeof(pixel_t)) + x];
    }
    // Sum the current warp
    size_t sum = warpReduceSum_sync(pixel);
    // And save the warp-total to shared memory
    if (lane == 0) {
        shared[warpId] = sum;
    }

    // Wait for all thread warps in the block to write
    __syncthreads();

    if (warpId == 0) {
        // Load the shared memory value for the warp corresponding to this lane.
        // We need to check this, because although we have a maximum number
        // of warps per block (32), we might have had less than that.
        sum = (lane < (blockDim.x * blockDim.y) / warpSize) ? shared[lane] : 0;
        // And sum all of the warps in this block together
        sum = warpReduceSum_sync(sum);
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
    assert(thread_block_size.x == 32);
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

    // And a device-side location to store results
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

        cudaMemcpy2D(dev_image,
                     device_pitch,
                     host_image.get(),
                     width * sizeof(pixel_t),
                     width * sizeof(pixel_t),
                     height,
                     cudaMemcpyHostToDevice);
        cuda_throw_error();

        do_sum_image<<<blocks_dims, thread_block_size>>>(
          dev_result, dev_image, device_pitch, width, height);

        cudaDeviceSynchronize();
        cuda_throw_error();

        // Copy the per-block sum data back, to sum CPU-side for now
        auto host_result =
          std::make_unique<std::remove_reference<decltype(*dev_result)>::type[]>(
            num_blocks);
        cudaMemcpy(host_result.get(),
                   dev_result,
                   sizeof(decltype(*dev_result)) * num_blocks,
                   cudaMemcpyDeviceToHost);
        cuda_throw_error();
        cudaDeviceSynchronize();

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

        print("\n");
    }
    cudaFree(dev_result);
    cudaFree(dev_image);
}
