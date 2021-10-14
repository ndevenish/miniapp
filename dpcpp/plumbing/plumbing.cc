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

// Constexpr power calculation
constexpr size_t cpow(size_t x, size_t power) {
    int ret = 1;
    for (int i = 0; i < power; ++i) {
        ret *= x;
    }
    return ret;
}

/// One-direction width of kernel. Total kernel span is (K_W * 2 + 1)
constexpr int KERNEL_WIDTH = 3;
/// One-direction height of kernel. Total kernel span is (K_W * 2 + 1)
constexpr int KERNEL_HEIGHT = 3;

constexpr int FULL_KERNEL_HEIGHT = KERNEL_HEIGHT * 2 + 1;

// Width of this array determines how many pixels we read at once
using PipedPixelsArray = std::array<H5Read::image_type, 16>;
// A convenience assignment for size of a single block
constexpr size_t BLOCK_SIZE = std::tuple_size<PipedPixelsArray>::value;
static_assert(is_power_of_two(BLOCK_SIZE));

// Convenience sum for PipedPixelsArray
auto operator+(const PipedPixelsArray& l, const PipedPixelsArray& r)
  -> PipedPixelsArray {
    PipedPixelsArray sum;
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        sum[i] = l[i] + r[i];
    }
    return sum;
}

// We need to buffer two blocks + kernel, because the pixels
// on the beginning of the block depend on the tail of the
// previous block, and the pixels at the end of the block
// depend on the start of the next block.
//
// Let's make a rolling buffer of:
//
//      | <KERNEL_WIDTH> | Block 0 | Block 1 |
//
// We read a block into block 1 - at which point we are
// ready to calculate all of the local-kernel sums for
// block 0 e.g.:
//
//      | K-2 | K-1 | k-0 | B0_0 | B0_1 | B0_2 | B0_3
//         └─────┴─────┴──────┼──────┴──────┴─────┘
//                            +
//                            │
//                         | S_0 | S_1 | S_2 | S_3 | ...
//
// Once we've calculated the per-pixel kernel sum for a
// single block, we can shift the entire array left by
// BLOCK_SIZE + KERNEL_WIDTH pixels to read the next
// block into the right of the buffer.
//
// Since we only need the raw pixel values of the
// buffer+block, this process can be pipelined.
using BufferedPipedPixelsArray =
  std::array<PipedPixelsArray::value_type, BLOCK_SIZE * 2 + KERNEL_WIDTH>;
// This two-block solution only works if kernel width < block size
static_assert(KERNEL_WIDTH < BLOCK_SIZE);

template <int blocks>
using ModuleRowStore = std::array<std::array<PipedPixelsArray, blocks>, KERNEL_HEIGHT>;

template <int id>
class ToModulePipe;

template <int id>
using ProducerPipeToModule = INTEL::pipe<class ToModulePipe<id>, PipedPixelsArray, 5>;

template <int Index>
class Module;

class Producer;

/// Return the profiling event time, in milliseconds, for an event
double event_ms(const sycl::event& e) {
    return 1e-6
           * (e.get_profiling_info<info::event_profiling::command_end>()
              - e.get_profiling_info<info::event_profiling::command_start>());
}

/// Calculate the GigaBytes per second given bytes, time
double GBps(size_t bytes, double ms) {
    return (static_cast<double>(bytes) / 1e9) / (ms / 1000.0);
}

/// Return value of event in terms of GigaBytes per second
double event_GBps(const sycl::event& e, size_t bytes) {
    const double ms = event_ms(e);
    return GBps(bytes, ms);
}

/// Calculate the prefix sum of a 2^N sized array
template <typename T, size_t BLOCK_SIZE>
void calculate_prefix_sum_inplace(std::array<T, BLOCK_SIZE>& data) {
    constexpr size_t BLOCK_SIZE_BITS = clog2(BLOCK_SIZE);
    static_assert(is_power_of_two(BLOCK_SIZE));

    // We need to store the last element to convert to inclusive
    auto last_element = data[BLOCK_SIZE - 1];

    // Parallel prefix scan - upsweep a binary tree
    // After this, every 1,2,4,8,... node has the correct
    // sum of the two nodes below it
#pragma unroll
    for (int d = 0; d < BLOCK_SIZE_BITS; ++d) {
#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k += cpow(2, d + 1)) {
            data[k + cpow(2, d + 1) - 1] =
              data[k + cpow(2, d) - 1] + data[k + cpow(2, d + 1) - 1];
        }
    }

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
// This calculated an exclusive sum. We want inclusive, so shift+add
#pragma unroll
    for (int i = 1; i < BLOCK_SIZE; ++i) {
        data[i - 1] = data[i];
    }
    data[BLOCK_SIZE - 1] = data[BLOCK_SIZE - 2] + last_element;
}

PipedPixelsArray sum_buffered_block_0(BufferedPipedPixelsArray& buffer) {
    // Now we can calculate the sums for block 0
    PipedPixelsArray sum{};
#pragma unroll
    for (int center = 0; center < BLOCK_SIZE; ++center) {
#pragma unroll
        for (int i = -KERNEL_WIDTH; i <= KERNEL_WIDTH; ++i) {
            sum[center] += buffer[KERNEL_WIDTH + center + i];
        }
    }
    return sum;
}

// /// Fallthrough for non-power-of-two arrays
// template <typename T, size_t size>
// void calculate_prefix_sum_inplace(std::array<T, size>& data) {
//     constexpr size_t constexpr size_t largest_pow2 = clog2(size);
//     calculate_prefix_sum_inplace

//     // fmt::print("Closest power of two for consolidating block sums: {} ({})\n",
//     //            fullw_pow2,
//     //            cpow(2, fullw_pow2));
//     // fmt::print("Remaining blocks: {}\n", FULL_BLOCKS - cpow(2, fullw_pow2));
// }

int main(int argc, char** argv) {
    auto start_time = std::chrono::high_resolution_clock::now();
    auto reader = H5Read(argc, argv);

    auto Q = initialize_queue();

    auto slow = reader.get_image_slow();
    auto fast = reader.get_image_fast();
    const size_t num_pixels = reader.get_image_slow() * reader.get_image_fast();

    // Mask data is the same for all images, so we copy it early
    auto mask_data = device_ptr<uint8_t>(malloc_device<uint8_t>(num_pixels, Q));
    auto image_data = host_ptr<uint16_t>(malloc_host<uint16_t>(num_pixels, Q));

    // PipedPixelsArray* result = malloc_host<PipedPixelsArray>(4, Q);
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
    constexpr size_t BLOCK_SIZE_BITS = clog2(BLOCK_SIZE);
    constexpr size_t BLOCK_REMAINDER = E2XE_16M_FAST % BLOCK_SIZE;
    constexpr size_t FULL_BLOCKS = (E2XE_16M_FAST - BLOCK_REMAINDER) / BLOCK_SIZE;

    // // Number of full blocks to cover all pixels in a module, including the edge
    // constexpr size_t BLOCKS_PER_MODULE =
    //   (E2XE_MOD_FAST / BLOCK_SIZE) + (E2XE_MOD_FAST % BLOCK_SIZE == 0 ? 0 : 1);

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

    constexpr size_t fullw_pow2 = clog2(FULL_BLOCKS);
    fmt::print("Closest power of two for consolidating block sums: {} ({})\n",
               fullw_pow2,
               cpow(2, fullw_pow2));
    fmt::print("Remaining blocks: {}\n", FULL_BLOCKS - cpow(2, fullw_pow2));

    std::array<uint16_t, FULL_BLOCKS>* totalblocksum =
      malloc_host<std::array<uint16_t, FULL_BLOCKS>>(slow, Q);

    uint16_t* destination_data = malloc_device<uint16_t>(num_pixels, Q);

    auto* rows_ptr = malloc_device<ModuleRowStore<FULL_BLOCKS>>(1, Q);
    // auto  rows = malloc_device<
    //                 //                        FULL_KERNEL_HEIGHT>{};

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
                          reinterpret_cast<PipedPixelsArray*>(image_data.get()
                                                              + y * fast));
                        ProducerPipeToModule<0>::write(image_data_h[block]);
                    }
                }
            });
        });

        // Launch a module kernel for every module
        event e_module = Q.submit([&](handler& h) {
            h.single_task<class Module<0>>([=](){
                auto result_h = host_ptr<PipedPixelsArray>(result);
                auto destination_data_d = device_ptr<uint16_t>(destination_data);

                size_t sum_pixels = 0;

                // Make a buffer for full rows so we can store them as we go
                // auto rows = std::array<std::array<PipedPixelsArray, FULL_BLOCKS>,
                //                        FULL_KERNEL_HEIGHT>{};
                auto rows = device_ptr<ModuleRowStore<FULL_BLOCKS>>(rows_ptr);

                for (size_t y = 0; y < slow; ++y) {
                    // The per-pixel buffer array to accumulate the blocks
                    BufferedPipedPixelsArray interim_pixels{};

                    // Have a "block" view of this pixel buffer for easy access
                    auto* interim_blocks = reinterpret_cast<PipedPixelsArray*>(
                      &interim_pixels[KERNEL_WIDTH]);

                    // Read the first block into initial position in the array
                    interim_blocks[0] = ProducerPipeToModule<0>::read();

                    for (size_t block = 0; block < FULL_BLOCKS - 1; ++block) {
                        // Read this into the right of the array...
                        interim_blocks[1] = ProducerPipeToModule<0>::read();

                        // Now we can calculate the sums for block 0
                        PipedPixelsArray sum = sum_buffered_block_0(interim_pixels);

                        // Now shift everything in the row buffer to the left
                        // to make room for the next pipe read
#pragma unroll
                        for (int i = 0; i < KERNEL_WIDTH + BLOCK_SIZE; ++i) {
                            interim_pixels[i] = interim_pixels[BLOCK_SIZE + i];
                        }

                        // Now we can insert this into the row accumulation store and
                        // do per-row calculations

                        // Calculate the new row - this is the integral sum
                        // of the previous FULL_KERNEL_HEIGHT rows, which
                        // we calculate by keeping a running total and
                        // subtracting the oldest row
                        PipedPixelsArray new_row;
                        auto prev_row = rows[0][0][block];
                        auto oldest_row = rows[0][FULL_KERNEL_HEIGHT - 1][block];

                        // Unrolling this causes II to raise to ~800
                        // #pragma unroll
                        for (int i = 0; i < BLOCK_SIZE; ++i) {
                            new_row[i] = sum[i] + prev_row[i] - oldest_row[i];
                        }

                        // #pragma unroll
                        // Shift all rows down to accomodate this new block

                        for (int i = 1; i < FULL_KERNEL_HEIGHT; ++i) {
                            rows[0][i][block] = rows[0][i - 1][block];
                        }
                        rows[0][0][block] = new_row;
                        // The new row - now stored in row 0 - is the "kernel sum"
                        // for the row (y - KERNEL_HEIGHT).
                        // copy into the destination block
                        // if (y >= KERNEL_HEIGHT) {
                        // for (int i = 0; i < BLOCK_SIZE; ++i) {
                        //     int row_y = y - KERNEL_HEIGHT;
                        //     row_y = row_y < 0 ? 0 : row_y;
                        //     size_t offset = row_y * fast + block * BLOCK_SIZE;
                        //     destination_data_d[offset + i] = new_row[i];
                        // }

                        *reinterpret_cast<PipedPixelsArray*>(
                          &destination_data_d[y * fast + block * BLOCK_SIZE]) = new_row;
                    }
                    // }
                    // Now, we have one last block - read zero into block 1 and sum it
                    // interim_blocks[1] = {};
                }
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

        // Copy the device destination buffer

        // Print a section of the image and "destination" arrays
        auto host_sum_data = host_ptr<uint16_t>(malloc_host<uint16_t>(num_pixels, Q));
        auto e_dest_download = Q.submit([&](handler& h) {
            h.memcpy(host_sum_data, destination_data, num_pixels * sizeof(uint16_t));
        });
        Q.wait();

        fmt::print("Data: ");
        for (int i = 0; i < fast; ++i) {
            fmt::print("{:3d}  ", image_data[i]);
        }
        fmt::print("\nSum:  ");
        for (int i = 0; i < fast; ++i) {
            fmt::print("{:3d}  ", host_sum_data[i]);
        }
        fmt::print("\n");
        // fmt::print("Out:   {}\n", result[0]);
        // fmt::print("Out2:  {}\n\n", result[1]);
        // fmt::print("In²:  {}\n", result[1]);
        // fmt::print("Out²: {}\n", result[3]);

        // fmt::print("\nTotal block sum:\n");
        // for (int i = slow - 5; i < slow; ++i) {
        //     fmt::print("{}\n", totalblocksum[i]);
        // }
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
