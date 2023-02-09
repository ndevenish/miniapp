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
#include <utility>

#include "common.hpp"
#include "h5read.h"

namespace cg = cooperative_groups;

using namespace fmt;

using pixel_t = H5Read::image_type;

/// One-direction width of kernel. Total kernel span is (K_W * 2 + 1)
constexpr int KERNEL_WIDTH = 3;
/// One-direction height of kernel. Total kernel span is (K_H * 2 + 1)
constexpr int KERNEL_HEIGHT = 3;

// namespace detail {

// template <typename T>
// void cudafree_delete(T *obj) {
//     cudaFree(obj);
// }

// }  // namespace detail

// cudaMalloc(&dev_result, sizeof(decltype(*dev_result)) * num_blocks);

template <typename T>
auto make_cuda_malloc(size_t num_items = 1) {
    T *obj = nullptr;
    if (cudaMalloc(&obj, sizeof(T) * num_items) != cudaSuccess || obj == nullptr) {
        throw std::bad_alloc{};
    }
    auto deleter = [](T *ptr) { cudaFree(ptr); };
    return std::unique_ptr<T, decltype(deleter)>{obj, deleter};
}

template <typename T>
auto make_cuda_managed_malloc(size_t num_items) {
    T *obj = nullptr;
    if (cudaMallocManaged(&obj, sizeof(T) * num_items) != cudaSuccess
        || obj == nullptr) {
        throw std::bad_alloc{};
    }
    auto deleter = [](T *ptr) { cudaFree(ptr); };
    return std::unique_ptr<T, decltype(deleter)>{obj, deleter};
}
/// Allocate memory using cudaMallocHost
template <typename T>
auto make_cuda_pinned_malloc(size_t num_items = 1) {
    T *obj = nullptr;
    if (cudaMallocHost(&obj, sizeof(T) * num_items) != cudaSuccess || obj == nullptr) {
        throw std::bad_alloc{};
    }
    auto deleter = [](T *ptr) { cudaFreeHost(ptr); };
    return std::unique_ptr<T, decltype(deleter)>{obj, deleter};
}

template <typename T>
auto make_cuda_pitched_malloc(size_t width, size_t height) {
    size_t pitch = 0;
    T *obj = nullptr;
    if (cudaMallocPitch(&obj, &pitch, width * sizeof(T), height) != cudaSuccess
        || obj == nullptr) {
        throw std::bad_alloc{};
    }
    auto deleter = [](T *ptr) { cudaFree(ptr); };
    return std::make_pair(std::unique_ptr<T[], decltype(deleter)>{obj, deleter},
                          pitch / sizeof(T));
}

class CudaEvent {
    cudaEvent_t event;

  public:
    CudaEvent() {
        if (cudaEventCreate(&event) != cudaSuccess) {
            cuda_throw_error();
        }
    }
    CudaEvent(cudaEvent_t event) : event(event) {}

    ~CudaEvent() {
        cudaEventDestroy(event);
    }
    void record(cudaStream_t stream = 0) {
        if (cudaEventRecord(event, stream) != cudaSuccess) {
            cuda_throw_error();
        }
    }
    /// Elapsed Event time, in milliseconds
    float elapsed_time(CudaEvent &since) {
        float elapsed_time = 0.0f;
        if (cudaEventElapsedTime(&elapsed_time, since.event, event) != cudaSuccess) {
            cuda_throw_error();
        }
        return elapsed_time;
    }
    void synchronize() {
        if (cudaEventSynchronize(event) != cudaSuccess) {
            cuda_throw_error();
        }
    }
};

template <typename T = uint8_t>
auto GBps(float time_ms, size_t number_objects) -> float {
    return 1000 * number_objects * sizeof(T) / time_ms / 1e9;
}

__global__ void do_spotfinding_naive(pixel_t *image,
                                     size_t image_pitch,
                                     uint8_t *mask,
                                     size_t mask_pitch,
                                     int width,
                                     int height,
                                     int *result_sum,
                                     size_t *result_sumsq,
                                     uint8_t *result_n,
                                     uint8_t *result_strong) {
    auto block = cg::this_thread_block();
    // auto warp = cg::tiled_partition<32>(block);
    // int warpId = warp.meta_group_rank();
    // int lane = warp.thread_rank();

    uint sum = 0;
    size_t sumsq = 0;
    uint8_t n = 0;

    int x = block.group_index().x * block.group_dim().x + block.thread_index().x;
    int y = block.group_index().y * block.group_dim().y + block.thread_index().y;

    for (int row = max(0, y - KERNEL_HEIGHT); row < min(y + KERNEL_HEIGHT + 1, height);
         ++row) {
        int row_offset = image_pitch * row;
        int mask_offset = mask_pitch * row;
        for (int col = max(0, x - KERNEL_WIDTH); col < min(x + KERNEL_WIDTH + 1, width);
             ++col) {
            pixel_t pixel = image[row_offset + col];
            uint8_t mask_pixel = mask[mask_offset + col];
            if (mask_pixel) {
                sum += pixel;
                sumsq += pixel * pixel;
                n += 1;
            }
        }
    }

    if (x < width && y < height) {
        result_sum[x + image_pitch * y] = sum;
        result_sumsq[x + image_pitch * y] = sumsq;
        result_n[x + mask_pitch * y] = n;
        result_strong[x + mask_pitch * y] = 0;
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

    // Create a host memory area to read the image into
    // auto host_image = std::make_unique<pixel_t[]>(width * height);
    auto host_image = make_cuda_pinned_malloc<pixel_t>(width * height);

    // Device-side pitched storage for image data
    auto [dev_image, device_pitch] = make_cuda_pitched_malloc<pixel_t>(width, height);
    auto [dev_mask, device_mask_pitch] =
      make_cuda_pitched_malloc<uint8_t>(width, height);
    print("Allocated device memory. Pitch = {} vs naive {}\n", device_pitch, width);

    // Managed memory areas for results
    auto result_sum = make_cuda_managed_malloc<int>(device_pitch * height);
    auto result_sumsq = make_cuda_managed_malloc<size_t>(device_pitch * height);
    auto result_n = make_cuda_managed_malloc<uint8_t>(device_mask_pitch * height);
    auto result_strong = make_cuda_managed_malloc<uint8_t>(device_mask_pitch * height);

    CudaEvent start, memcpy, kernel, all;

    size_t mask_sum = 0;
    if (reader.get_mask()) {
        mask_sum = 0;
        for (size_t i = 0; i < width * height; ++i) {
            if (reader.get_mask().value()[i]) {
                mask_sum += 1;
            }
        }
        start.record();
        cudaMemcpy2D(dev_mask.get(),
                     device_mask_pitch,
                     reader.get_mask()->data(),
                     width,
                     width,
                     height,
                     cudaMemcpyHostToDevice);
        cuda_throw_error();
    } else {
        mask_sum = width * height;
        start.record();
        cudaMemset(dev_mask.get(), 1, device_mask_pitch * height);
        cuda_throw_error();
    }
    memcpy.record();
    memcpy.synchronize();

    float memcpy_time = memcpy.elapsed_time(start);
    print("Uploaded mask ({:.2f} Mpx) in {:.2f} ms ({:.1f} GBps)\n",
          static_cast<float>(mask_sum) / 1e6,
          memcpy_time / 1000,
          GBps(memcpy_time, width * height));

    print("\nProcessing Images\n\n");

    for (size_t image_id = 0; image_id < reader.get_number_of_images(); ++image_id) {
        print("Image {}:\n", image_id);
        reader.get_image_into(image_id, host_image.get());

        // Copy data to GPU
        // Copy the image to GPU
        start.record();
        cudaMemcpy2D(dev_image.get(),
                     device_pitch * sizeof(pixel_t),
                     host_image.get(),
                     width * sizeof(pixel_t),
                     width * sizeof(pixel_t),
                     height,
                     cudaMemcpyHostToDevice);
        memcpy.record();
        cudaDeviceSynchronize();
        cuda_throw_error();

        do_spotfinding_naive<<<blocks_dims, thread_block_size>>>(dev_image.get(),
                                                                 device_pitch,
                                                                 dev_mask.get(),
                                                                 device_mask_pitch,
                                                                 width,
                                                                 height,
                                                                 result_sum.get(),
                                                                 result_sumsq.get(),
                                                                 result_n.get(),
                                                                 result_strong.get());
        kernel.record();
        all.record();
        cuda_throw_error();
        cudaDeviceSynchronize();

        print("  Upload Time: \033[1m{:5.2f}\033[0m ms \033[37m({:.1f} GBps)\033[0m\n",
              memcpy.elapsed_time(start),
              GBps<pixel_t>(memcpy.elapsed_time(start), width * height));
        print("  Kernel Time: \033[1m{:5.2f}\033[0m ms\n", kernel.elapsed_time(memcpy));
        print("               ════════\n");
        print("        Total: \033[1m{:5.2f}\033[0m ms ({:.1f} GBps)\n",
              all.elapsed_time(start),
              GBps<pixel_t>(all.elapsed_time(start), width * height));

        int strong = 0;
        for (size_t row = 0; row < height; ++row) {
            for (size_t col = 0; col < width; ++col) {
                if (result_strong.get()[row * device_mask_pitch + col]) {
                    strong += 1;
                }
            }
        }
        print("       Strong: {} px\n", strong);
        draw_image_data(result_sum.get(), 0, 0, 15, 15, device_pitch, height);
        draw_image_data(result_n.get(), 0, 0, 15, 15, device_mask_pitch, height);
        print("\n");
    }
}
