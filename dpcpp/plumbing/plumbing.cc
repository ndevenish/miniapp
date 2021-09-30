#include <fmt/color.h>
#include <fmt/core.h>
#include <fmt/ranges.h>

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <iostream>

#include "common.hpp"
#include "eiger2xe.h"
#include "h5read.h"

using namespace sycl;

template <int id>
class ToModulePipe;

// From https://stackoverflow.com/a/10585543/1118662
constexpr bool is_power_of_two(int x) {
    return x && ((x & (x - 1)) == 0);
}

/// constexpr flooring log2
constexpr size_t clog2(size_t n) {
    size_t result = 0;
    while (n >= 2) {
        result += 1;
        n = n / 2;
    }
    return result;
}

constexpr size_t cpow(size_t x, size_t power) {
    int ret = 1;
    for (int i = 0; i < power; ++i) {
        ret *= x;
    }
    return ret;
    // return power == 0 ? 1 : x * pow(x, power - 1);
}

// Width of this array determines how many pixels we read at once
using PipedPixelsArray = std::array<H5Read::image_type, 16>;
static_assert(is_power_of_two(std::tuple_size<PipedPixelsArray>::value));

template <int id>
using ProducerPipeToModule = INTEL::pipe<class ToModulePipe<id>, PipedPixelsArray, 5>;

template <int Index>
class Module;

class Producer;

double event_ms(const sycl::event& e) {
    return 1e-6
           * (e.get_profiling_info<info::event_profiling::command_end>()
              - e.get_profiling_info<info::event_profiling::command_start>());
}

double GBps(size_t bytes, double ms) {
    return (static_cast<double>(bytes) / 1e9) / (ms / 1000.0);
}

/// Return value of event in terms of GigaBytes per second
double event_GBps(const sycl::event& e, size_t bytes) {
    const double ms = event_ms(e);
    return GBps(bytes, ms);
}

/// Calculate the prefix sum of a PipedPixelsArray
void calculate_prefix_sum_inplace(PipedPixelsArray& data) {
    // Parallel prefix scan - upsweep a binary tree
    // After this, every 1,2,4,8,... node has the correct
    // sum of the two nodes below it

    constexpr size_t BLOCK_SIZE = std::tuple_size<PipedPixelsArray>::value;
    constexpr size_t BLOCK_SIZE_BITS = clog2(BLOCK_SIZE);

#pragma unroll
    for (int d = 0; d < BLOCK_SIZE_BITS; ++d) {
#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k += cpow(2, d + 1)) {
            data[k + cpow(2, d + 1) - 1] =
              data[k + cpow(2, d) - 1] + data[k + cpow(2, d + 1) - 1];
        }
    }
    // Total sum for this block is now in data[-1]

    // Parallel prefix downsweep the binary tree
    // After this, entire block has the correct prefix sum
    data[BLOCK_SIZE - 1] = 0;
#pragma unroll
    for (int d = BLOCK_SIZE_BITS - 1; d >= 0; --d) {
#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k += cpow(2, d + 1)) {
            // Save the left node value
            auto t = data[k + cpow(2, d) - 1];
            // Left node becomes parent
            data[k + cpow(2, d) - 1] = data[k + cpow(2, d + 1) - 1];
            // Right node becomes root + previous left value
            data[k + cpow(2, d + 1) - 1] += t;
        }
    }
}

int main(int argc, char** argv) {
    auto start_time = std::chrono::high_resolution_clock::now();
    auto reader = H5Read(argc, argv);

    auto Q = initialize_queue();

    auto slow = reader.get_image_slow();
    auto fast = reader.get_image_fast();
    const size_t num_pixels = reader.get_image_slow() * reader.get_image_fast();

    // Mask data is the same for all images, so we copy it early
    uint8_t* mask_data = malloc_device<uint8_t>(num_pixels, Q);
    uint16_t* image_data = malloc_host<uint16_t>(num_pixels, Q);
    PipedPixelsArray* result = malloc_host<PipedPixelsArray>(2, Q);
    assert(image_data % 64 == 0);
    fmt::print("Uploading mask data to accelerator.... ");
    auto e_mask_upload = Q.submit(
      [&](handler& h) { h.memcpy(mask_data, reader.get_mask().data(), num_pixels); });
    Q.wait();
    fmt::print("done in {:.1f} ms ({:.2f} GBps)\n",
               event_ms(e_mask_upload),
               event_GBps(e_mask_upload, num_pixels));

    // Module/detector compile-time calculations
    constexpr size_t BLOCK_SIZE = std::tuple_size<PipedPixelsArray>::value;
    constexpr size_t BLOCK_SIZE_BITS = clog2(BLOCK_SIZE);
    constexpr size_t BLOCK_REMAINDER = E2XE_16M_FAST % BLOCK_SIZE;
    constexpr size_t FULL_BLOCKS = (E2XE_16M_FAST - BLOCK_REMAINDER) / BLOCK_SIZE;
    // constexpr size_t TOTAL_BLOCKS_UNALIGNED =
    //   (E2XE_16M_FAST * E2XE_16M_SLOW) / BLOCK_SIZE;
    static_assert(is_power_of_two(BLOCK_SIZE));
    fmt::print(
      "Block data:\n           SIZE: {} px per block\n           BITS: {} bits to "
      "store \n      REMAINDER: "
      "{} px unprocessed per row\n    FULL_BLOCKS: {} blocks across image width\n",
      BLOCK_SIZE,
      BLOCK_SIZE_BITS,
      BLOCK_REMAINDER,
      FULL_BLOCKS);

    fmt::print("Starting image loop:\n");
    for (int i = 0; i < reader.get_number_of_images(); ++i) {
        fmt::print("\nReading Image {}\n", i);
        reader.get_image_into(i, image_data);

        fmt::print("Calculating host sum\n");
        // Now we are using blocks and discarding excess, do that here
        size_t host_sum = 0;
        for (int i = 0; i < FULL_BLOCKS * BLOCK_SIZE; ++i) {
            host_sum += image_data[i];
        }
        fmt::print("Starting Kernels\n");
        auto t1 = std::chrono::high_resolution_clock::now();

        event e_producer = Q.submit([&](handler& h) {
            h.single_task<class Producer>([=]() {
                // For now, send every pixel into one pipe
                // We are using blocks based on the pipe width - this is
                // likely not an exact divisor of the fast width, so for
                // now just ignore the excess pixels
                for (size_t y = 0; y < slow; ++y) {
                    for (size_t block = 0; block < FULL_BLOCKS; ++block) {
                        auto image_data_h = host_ptr<PipedPixelsArray>(
                          reinterpret_cast<PipedPixelsArray*>(image_data + y * fast));
                        ProducerPipeToModule<0>::write(image_data_h[block]);
                    }
                }
            });
        });

        // Launch a module kernel for every module
        event e_module = Q.submit([&](handler& h) {
            h.single_task<class Module<0>>([=](){
                size_t sum_pixels = 0;

                // for (size_t y = 0; y < slow; ++y) {
                // We have evenly sized blocks send to us
                for (size_t block = 0; block < FULL_BLOCKS * slow; ++block) {
                    auto result_h = host_ptr<PipedPixelsArray>(result);
                    PipedPixelsArray sum = ProducerPipeToModule<0>::read();
                    result_h[0] = sum;

                    calculate_prefix_sum_inplace(sum);
                    result_h[1] = sum;
                }
                // }
            });
        });

        Q.wait();
        auto t2 = std::chrono::high_resolution_clock::now();
        double ms_all =
          std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count()
          * 1000;

        fmt::print(" ... produced in {:.2f} ms ({:.3g} GBps)\n",
                   event_ms(e_producer),
                   event_GBps(e_producer, num_pixels * sizeof(uint16_t) / 2));
        fmt::print(" ... consumed in {:.2f} ms ({:.3g} GBps)\n",
                   event_ms(e_module),
                   event_GBps(e_module, num_pixels * sizeof(uint16_t) / 2));
        fmt::print(" ... Total consumed + piped in host time {:.2f} ms ({:.3g} GBps)\n",
                   ms_all,
                   GBps(num_pixels * sizeof(uint16_t), ms_all));

        // auto device_sum = result[0];
        // auto color = fg(host_sum == device_sum ? fmt::color::green :
        // fmt::color::red); fmt::print(color, "     Sum = {} / {}\n", device_sum, host_sum);
        fmt::print("In:  {}\n", result[0]);
        fmt::print("Out: {}\n", result[1]);
    }

    free(result, Q);
    free(image_data, Q);
    free(mask_data, Q);
    auto end_time = std::chrono::high_resolution_clock::now();

    fmt::print(
      "Total run duration: {:.2f} s\n",
      std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time)
        .count());
}
