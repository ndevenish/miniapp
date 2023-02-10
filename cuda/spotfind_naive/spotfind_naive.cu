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
#include <chrono>
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

    // Don't calculate for masked pixels
    bool px_is_valid = mask[y * mask_pitch + x] != 0;
    pixel_t this_pixel = image[y * image_pitch + x];

    if (px_is_valid) {
        for (int row = max(0, y - KERNEL_HEIGHT);
             row < min(y + KERNEL_HEIGHT + 1, height);
             ++row) {
            int row_offset = image_pitch * row;
            int mask_offset = mask_pitch * row;
            for (int col = max(0, x - KERNEL_WIDTH);
                 col < min(x + KERNEL_WIDTH + 1, width);
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
    }

    if (x < width && y < height) {
        result_sum[x + image_pitch * y] = sum;
        result_sumsq[x + image_pitch * y] = sumsq;
        result_n[x + mask_pitch * y] = n;

        // if (mask[y * mask_pitch + x]) {

        // Calculate the thresholding
        if (px_is_valid) {
            constexpr float n_sig_s = 3.0f;
            constexpr float n_sig_b = 6.0f;

            float mean = sum / n;
            float variance = (n * sumsq - (sum * sum)) / (n * (n - 1));
            float dispersion = variance / mean;
            bool not_background = dispersion > n_sig_b;
            float signal_threshold = mean + n_sig_s * sqrt(mean);
            bool is_signal = this_pixel > signal_threshold;
            bool is_strong_pixel = not_background && is_signal;
            result_strong[x + mask_pitch * y] = is_strong_pixel;
            // double a = m * y - x * x - x * (m - 1);
            // double b = m * src[k] - x;
            // double c = x * nsig_b_ * std::sqrt(2 * (m - 1));
            // double d = nsig_s_ * std::sqrt(x * m);
            // dst[k] = a > c && b > d;
            /*

                            float mean = sum / N;
                            auto variance = (N * sum_sq - (sum * sum)) / (N * (N - 1));
                            // std::array<float, BLOCK_SIZE> variance;
                            // for (size_t i = 0; i < BLOCK_SIZE; ++i) {
                            // float variance = static_cast<float>(
                            //   (static_cast<float>(N)
                            //      * static_cast<float>(kernel_sum_sq[i])
                            //    - static_cast<float>(kernel_sum[i])
                            //        * static_cast<float>(kernel_sum[i]))
                            //   / (static_cast<float>(N) * (static_cast<float>(N) - 1)));
                            // }

                            auto dispersion = variance / mean;
                            auto is_background = dispersion > background_threshold;

                            auto signal_threshold = mean + sigma_strong * sqrt(mean);
                            auto is_signal = kernel_px[i] > signal_threshold;

                            auto is_strong_pixel = is_background && is_signal;

                            if (is_strong_pixel) {
                                _count += 1;
                            }
            */
        } else {
            result_strong[x + mask_pitch * y] = 0;
        }
    }
}

/** Calculate a kernel sum, the simplest possible implementation.
 * 
 * This is **slow**, even from the perspective of something running
 * infrequently. It's relatively simple to get the algorithm correct,
 * however, so is useful for validating other algorithms.
 **/
template <typename Tin, typename Tmask>
auto calculate_kernel_sum_slow(Tin *data,
                               Tmask *mask,
                               std::size_t fast,
                               std::size_t slow) {
    auto out_d = std::make_unique<size_t[]>(slow * fast);
    size_t *out = out_d.get();
    for (int y = 0; y < slow; ++y) {
        int y0 = std::max(0, y - KERNEL_HEIGHT);
        int y1 = std::min((int)slow, y + KERNEL_HEIGHT + 1);
        // std::size_t y1 = std::min((int)slow-1, y + KERNEL_HEIGHT);
        for (int x = 0; x < fast; ++x) {
            int x0 = std::max(0, x - KERNEL_WIDTH);
            int x1 = std::min((int)fast, x + KERNEL_WIDTH + 1);
            size_t acc{};
            for (int ky = y0; ky < y1; ++ky) {
                for (int kx = x0; kx < x1; ++kx) {
                    if (mask[ky * fast + kx]) {
                        acc += data[(ky * fast) + kx];
                    }
                }
            }
            out[(y * fast) + x] = mask[y * fast + x] ? acc : 0;
        }
    }
    return out_d;
}

/** Calculate a kernel sum, on the host, using an SAT.
 * 
 * This is designed for non-offloading calculations for e.g. crosscheck
 * or precalculation (like the mask).
 **/
template <typename Tin, typename Tmask, typename Taccumulator = Tin>
auto calculate_kernel_sum_sat(Tin *data,
                              Tmask *mask,
                              std::size_t fast,
                              std::size_t slow) -> std::unique_ptr<Tin[]> {
    auto sat_d = std::make_unique<Taccumulator[]>(slow * fast);
    Taccumulator *sat = sat_d.get();
    for (int y = 0; y < slow; ++y) {
        Taccumulator acc = 0;
        for (int x = 0; x < fast; ++x) {
            if (mask[y * fast + x]) {
                acc += data[y * fast + x];
            }
            if (y == 0) {
                sat[y * fast + x] = acc;
            } else {
                sat[y * fast + x] = acc + sat[(y - 1) * fast + x];
            }
        }
    }

    // Now evaluate the (fixed size) kernel across this SAT
    auto out_d = std::make_unique<Tin[]>(slow * fast);
    Tin *out = out_d.get();
    for (int y = 0; y < slow; ++y) {
        int y0 = y - KERNEL_HEIGHT - 1;
        int y1 = std::min((int)slow - 1, y + KERNEL_HEIGHT);
        for (int x = 0; x < fast; ++x) {
            int x0 = x - KERNEL_WIDTH - 1;
            int x1 = std::min((int)fast - 1, x + KERNEL_WIDTH);

            Taccumulator tl{}, tr{}, bl{};
            Taccumulator br = sat[y1 * fast + x1];

            if (y0 >= 0 && x0 >= 0) {
                // Top left fully inside kernel
                tl = sat[y0 * fast + x0];
                tr = sat[y0 * fast + x1];
                bl = sat[y1 * fast + x0];
            } else if (x0 >= 0) {
                // Top rows - y0 outside range
                bl = sat[y1 * fast + x0];
            } else if (y0 >= 0) {
                // Left rows - x0 outside range
                tr = sat[y0 * fast + x1];
            }
            out[y * fast + x] = br - (bl + tr) + tl;
        }
    }

    return out_d;
}

template <typename T, typename U>
bool compare_results(const T *left,
                     const size_t left_pitch,
                     const U *right,
                     const size_t right_pitch,
                     std::size_t width,
                     std::size_t height,
                     size_t *mismatch_x = nullptr,
                     size_t *mismatch_y = nullptr) {
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            T lval = left[left_pitch * y + x];
            U rval = right[right_pitch * y + x];
            if (lval != rval) {
                if (mismatch_x != nullptr) {
                    *mismatch_x = x;
                }
                if (mismatch_y != nullptr) {
                    *mismatch_y = y;
                }
                print("First mismatch at ({}, {}), Left {} != {} Right\n",
                      x,
                      y,
                      lval,
                      rval);
                return false;
            }
        }
    }
    return true;
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
          memcpy_time,
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

        print("  Upload Time: \033[1m{:6.2f}\033[0m ms \033[37m({:.1f} GBps)\033[0m\n",
              memcpy.elapsed_time(start),
              GBps<pixel_t>(memcpy.elapsed_time(start), width * height));
        print("  Kernel Time: \033[1m{:6.2f}\033[0m ms\n", kernel.elapsed_time(memcpy));
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

        // Calculate on CPU to compare
        auto start_time = std::chrono::high_resolution_clock::now();
        auto validate_sum = calculate_kernel_sum_slow(
          host_image.get(), reader.get_mask().value().data(), width, height);

        draw_image_data(result_sum, 0, 0, 15, 15, device_pitch, height);
        // draw_image_data(result_n, 0, 0, 15, 15, device_mask_pitch, height);

        size_t mismatch_x = 0, mismatch_y = 0;
        bool validation_matches = compare_results(validate_sum.get(),
                                                  width,
                                                  result_sum.get(),
                                                  device_pitch,
                                                  width,
                                                  height,
                                                  &mismatch_x,
                                                  &mismatch_y);
        auto end_time = std::chrono::high_resolution_clock::now();
        float validation_time =
          std::chrono::duration_cast<std::chrono::duration<double>>(end_time
                                                                    - start_time)
            .count()
          * 1000;

        if (validation_matches) {
            print("     Compared: \033[32mMatch\033[0m in {:.0f} ms\n",
                  validation_time);
        } else {
            print("     Compared: \033[1;31mMismatch\033[0m in {:.0f} ms\n",
                  validation_time);
            mismatch_x = max(static_cast<int>(mismatch_x) - 8, 0);
            mismatch_y = max(static_cast<int>(mismatch_y) - 8, 0);
            print("From Validator:\n");
            draw_image_data(
              validate_sum.get(), mismatch_x, mismatch_y, 16, 16, width, height);
            print("From kernel:\n");
            draw_image_data(
              result_sum, mismatch_x, mismatch_y, 16, 16, device_pitch, height);
            print("Resultant N:\n");

            draw_image_data(
              result_n, mismatch_x, mismatch_y, 16, 16, device_mask_pitch, height);
            print("Mask:\n");

            draw_image_data(reader.get_mask().value().data(),
                            mismatch_x,
                            mismatch_y,
                            16,
                            16,
                            width,
                            height);
        }

        print("\n");
    }
}
