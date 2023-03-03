/**
 * Basic Naive Kernel
 * 
 * Does spotfinding in-kernel, without in-depth performance tweaking.
 * 
 */

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <fmt/core.h>

#include <array>
#include <cassert>
#include <chrono>
#include <memory>
#include <utility>

#include "common.hpp"
#include "h5read.h"
#include "standalone.h"

namespace cg = cooperative_groups;

using namespace fmt;

using pixel_t = H5Read::image_type;

/// One-direction width of kernel. Total kernel span is (K_W * 2 + 1)
constexpr int KERNEL_WIDTH = 3;
/// One-direction height of kernel. Total kernel span is (K_H * 2 + 1)
constexpr int KERNEL_HEIGHT = 3;

template <typename T, typename Sum = T>
__device__ auto calculate_area_sum(T exchange_block[32][32],
                                   const cg::thread_block &block) -> Sum {
    Sum sum = 0;
    // If we aren't in the edge KERNEL pixels, then we calculate and update
    if (block.thread_index().x >= KERNEL_WIDTH
        && block.thread_index().y >= KERNEL_HEIGHT
        && block.thread_index().x < block.dim_threads().x - KERNEL_WIDTH
        && block.thread_index().y < block.dim_threads().y - KERNEL_HEIGHT) {
        // Central block x,y coordinates
        const int bX = block.thread_index().x;
        const int bY = block.thread_index().y;

        // Locations of four corners for SAT
        const int l = bX - KERNEL_WIDTH;
        const int r = bX + KERNEL_WIDTH + 1;
        const int t = bY - KERNEL_HEIGHT;
        const int b = bY + KERNEL_HEIGHT + 1;

        // Reading out these coordinates
        int A = exchange_block[b][r];
        int B = exchange_block[t][r];
        int C = exchange_block[b][l];
        int D = exchange_block[t][l];

        sum = A - B - C + D;
    }
    return sum;
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
    __shared__ uint64_t block_exchange_8[32][32];
    __shared__ uint32_t block_exchange_4[32][32];
    __shared__ uint16_t block_exchange_2[32][32];

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    // int warpId = warp.meta_group_rank();
    // int lane = warp.thread_rank();

    uint sum = 0;
    size_t sumsq = 0;
    uint8_t n = 0;

    // The target image pixel of this thread
    int x = block.group_index().x * block.group_dim().x + block.thread_index().x
            - KERNEL_WIDTH;
    int y = block.group_index().y * block.group_dim().y + block.thread_index().y
            - KERNEL_HEIGHT;

    // Make sure this pixel isn't masked or off-image
    bool px_is_valid = false;
    pixel_t this_pixel = 0;
    if (x >= 0 && y >= 0 && x < width && y < height) [[likely]] {
        px_is_valid = mask[y * mask_pitch + x] != 0;
        this_pixel = image[y * image_pitch + x];
    }
    size_t this_pixel_sq = this_pixel * this_pixel;

    // Calculate the horizontal prefix sums
    auto inc_sum = cg::exclusive_scan(warp, this_pixel);
    auto inc_sumsq = cg::exclusive_scan(warp, this_pixel_sq);
    auto inc_n = cg::exclusive_scan(warp, px_is_valid ? 1 : 0);

    // Broadcast this value to the rest of the block
    block_exchange_8[block.thread_index().y][block.thread_index().x] = inc_sumsq;
    block_exchange_4[block.thread_index().y][block.thread_index().x] = inc_sum;
    block_exchange_2[block.thread_index().y][block.thread_index().x] = inc_n;

    // Wait until we can do block-level exchanges
    __syncthreads();

    // Transpose the block
    inc_sumsq = block_exchange_8[block.thread_index().x][block.thread_index().y];
    inc_sum = block_exchange_4[block.thread_index().x][block.thread_index().y];
    inc_n = block_exchange_2[block.thread_index().x][block.thread_index().y];

    inc_sumsq = cg::exclusive_scan(warp, inc_sumsq);
    inc_sum = cg::exclusive_scan(warp, inc_sum);
    inc_n = cg::exclusive_scan(warp, inc_n);

    // And, write it back
    block_exchange_8[block.thread_index().y][block.thread_index().x] = inc_sumsq;
    block_exchange_4[block.thread_index().y][block.thread_index().x] = inc_sum;
    block_exchange_2[block.thread_index().y][block.thread_index().x] = inc_n;

    __syncthreads();

    sumsq = calculate_area_sum(block_exchange_8, block);
    sum = calculate_area_sum(block_exchange_4, block);
    n = calculate_area_sum(block_exchange_2, block);

    // if (x >= 0 && y >= 0 && x < width && y < height) {
    //     // // Pull down the incremental sum for this pixel again so we can write to global
    //     // inc_sumsq = block_exchange_8[block.thread_index().y][block.thread_index().x];
    //     // inc_sum = block_exchange_4[block.thread_index().y][block.thread_index().x];
    //     // inc_n = block_exchange_2[block.thread_index().y][block.thread_index().x];
    //     result_sum[y * image_pitch + x] = sum;
    //     result_sumsq[y * image_pitch + x] = sumsq;
    //     result_n[y * mask_pitch + x] = n;
    // }

    if (x < width && y < height && x >= 0 && y >= 0) {
        // Calculate the thresholding
        if (px_is_valid) {
            constexpr float n_sig_s = 3.0f;
            constexpr float n_sig_b = 6.0f;

            float sum_f = static_cast<float>(sum);
            float sumsq_f = static_cast<float>(sumsq);

            float mean = sum_f / n;
            float variance = (n * sumsq_f - (sum_f * sum_f)) / (n * (n - 1));
            float dispersion = variance / mean;
            float background_threshold = 1 + n_sig_b * sqrt(2.0f / (n - 1));
            bool not_background = dispersion > background_threshold;
            float signal_threshold = mean + n_sig_s * sqrt(mean);
            bool is_signal = this_pixel > signal_threshold;
            bool is_strong_pixel = not_background && is_signal;
            result_strong[x + mask_pitch * y] = is_strong_pixel;
        } else {
            result_strong[x + mask_pitch * y] = 0;
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
    dim3 thread_block_size{32, 32};
    // Make enough blocks to overlap the edges with the kernel
    dim3 blocks_dims{
      static_cast<unsigned int>(
        ceilf(static_cast<float>(width + KERNEL_WIDTH * 2) / thread_block_size.x)),
      static_cast<unsigned int>(
        ceilf(static_cast<float>(height + KERNEL_HEIGHT * 2) / thread_block_size.y))};
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
    // Make sure to clear these completely
    cudaMemset(result_sum.get(), 0, sizeof(int) * device_pitch * height);
    cudaMemset(result_sumsq.get(), 0, sizeof(size_t) * device_pitch * height);
    cudaMemset(result_n.get(), 0, sizeof(uint8_t) * device_mask_pitch * height);
    cudaMemset(result_strong.get(), 0, sizeof(uint8_t) * device_mask_pitch * height);
    cudaDeviceSynchronize();
    cuda_throw_error();

    CudaEvent pre_load, start, memcpy, kernel, all;

    size_t mask_sum = 0;
    if (reader.get_mask()) {
        mask_sum = 0;
        for (size_t i = 0; i < width * height; ++i) {
            if (reader.get_mask().value()[i]) {
                mask_sum += 1;
            }
        }
        start.record();
        cudaMemcpy2DAsync(dev_mask.get(),
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

    print("\nProcessing {} Images\n\n", reader.get_number_of_images());
    auto spotfinder = StandaloneSpotfinder(width, height);

    for (size_t image_id = 0; image_id < reader.get_number_of_images(); ++image_id) {
        if (args.image_number.has_value() && args.image_number.value() != image_id) {
            continue;
        }

        print("Image {}:\n", image_id);
        pre_load.record();
        pre_load.synchronize();

        reader.get_image_into(image_id, host_image.get());
        host_image.get()[10 + width * 10] = 50;

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

        print("    Read Time: \033[1m{:6.2f}\033[0m ms \033[37m{:>11}\033[0m\n",
              start.elapsed_time(pre_load),
              format("({:4.1f} GBps)",
                     GBps<pixel_t>(start.elapsed_time(pre_load), width * height)));
        print("  Upload Time: \033[1m{:6.2f}\033[0m ms \033[37m({:4.1f} GBps)\033[0m\n",
              memcpy.elapsed_time(start),
              GBps<pixel_t>(memcpy.elapsed_time(start), width * height));
        print("  Kernel Time: \033[1m{:6.2f}\033[0m ms \033[37m{:>11}\033[0m\n",
              kernel.elapsed_time(memcpy),
              format("({:.1f} GBps)",
                     GBps<pixel_t>(kernel.elapsed_time(memcpy), width * height)));
        print("               ════════\n");
        print("        Total: \033[1m{:6.2f}\033[0m ms {:>11}\n",
              all.elapsed_time(pre_load),
              format("({:.1f} GBps)",
                     GBps<pixel_t>(all.elapsed_time(pre_load), width * height)));
        auto strong =
          count_nonzero(result_strong.get(), width, height, device_mask_pitch);
        print("       Strong: {} px\n", strong);

        auto start_time = std::chrono::high_resolution_clock::now();
        size_t mismatch_x = 0, mismatch_y = 0;

        auto converted_image =
          std::vector<double>{host_image.get(), host_image.get() + width * height};
        auto dials_strong = spotfinder.standard_dispersion(
          converted_image, reader.get_mask().value_or(span<uint8_t>{}));
        auto end_time = std::chrono::high_resolution_clock::now();
        size_t dials_results = count_nonzero(dials_strong, width, height, width);

        float validation_time_ms =
          std::chrono::duration_cast<std::chrono::duration<double>>(end_time
                                                                    - start_time)
            .count()
          * 1000;
        print("        Dials: {} px in {:.0f} ms CPU time\n",
              dials_results,
              validation_time_ms);

        bool validation_matches = compare_results(dials_strong.data(),
                                                  width,
                                                  result_strong.get(),
                                                  device_mask_pitch,
                                                  width,
                                                  height,
                                                  &mismatch_x,
                                                  &mismatch_y);

        // if (validation_matches) {
        //     print("     Compared: \033[32mMatch\033[0m\n");
        // } else {
        //     print("     Compared: \033[1;31mMismatch\033[0m\n");
        //     mismatch_x = max(static_cast<int>(mismatch_x) - 8, 0);
        //     mismatch_y = max(static_cast<int>(mismatch_y) - 8, 0);
        //     print("Data:\n");
        //     draw_image_data(host_image, mismatch_x, mismatch_y, 16, 16, width, height);
        //     print("Strong From DIALS:\n");
        //     draw_image_data(
        //       dials_strong, mismatch_x, mismatch_y, 16, 16, width, height);
        //     print("Strong From kernel:\n");
        //     draw_image_data(
        //       result_strong, mismatch_x, mismatch_y, 16, 16, device_mask_pitch, height);
        //     // print("Resultant N:\n");
        //     print("Sum From kernel:\n");
        //     draw_image_data(
        //       result_sum, mismatch_x, mismatch_y, 16, 16, device_pitch, height);
        //     print("Sum² From kernel:\n");
        //     draw_image_data(
        //       result_sumsq, mismatch_x, mismatch_y, 16, 16, device_pitch, height);
        //     print("Mask:\n");
        //     draw_image_data(reader.get_mask().value().data(),
        //                     mismatch_x,
        //                     mismatch_y,
        //                     16,
        //                     16,
        //                     width,
        //                     height);
        // }
        print("Image:\n");
        draw_image_data(host_image, 0, 0, 25, 16, width, height);
        print("Sum:\n");
        draw_image_data(result_sum, 0, 0, 25, 16, device_pitch, height);
        print("SumSq:\n");
        draw_image_data(result_sumsq, 0, 0, 25, 16, device_pitch, height);
        print("N:\n");
        draw_image_data(result_n, 0, 0, 25, 16, device_mask_pitch, height);

        print("\n\n");
    }
}
