// #include <span>

#include "kernel.hpp"

#include <array>
#include <span.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "common.hpp"
#include "fmt/core.h"

using namespace sycl;

class ToModulePipe;

using ProducerPipeToModule = SYCL_INTEL::pipe<class ToModulePipe, PipedPixelsArray, 5>;

class Module;
class Producer;

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
//   PipedPixelsArray::value_type[BLOCK_SIZE * 2 + KERNEL_WIDTH];
// This two-block solution only works if kernel width < block size
static_assert(KERNEL_WIDTH < BLOCK_SIZE);

template <int blocks>
using ModuleRowStore = PipedPixelsArray[FULL_KERNEL_HEIGHT][blocks];

// Convenience operators for PipedPixelsArray
inline auto operator+(const PipedPixelsArray& l, const PipedPixelsArray& r)
  -> PipedPixelsArray {
    PipedPixelsArray sum;
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        sum.data[i] = l.data[i] + r.data[i];
    }
    return sum;
}
inline auto operator-(const PipedPixelsArray& l, const PipedPixelsArray& r)
  -> PipedPixelsArray {
    PipedPixelsArray sum;
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        sum.data[i] = l.data[i] - r.data[i];
    }
    return sum;
}

auto pow2(const PipedPixelsArray& val) -> PipedPixelsArray {
    PipedPixelsArray sqr;
#pragma unroll
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        sqr[i] = val[i] * val[i];
    }
    return sqr;
}

const sycl::stream& operator<<(const sycl::stream& os,
                               const BufferedPipedPixelsArray& obj) {
    os << "[ ";
    for (int i = 0; i < KERNEL_WIDTH; ++i) {
        if (i != 0) os << ", ";
        os << setw(2) << obj[i];
    }
    os << " | ";
    for (int i = KERNEL_WIDTH; i < BLOCK_SIZE + KERNEL_WIDTH; ++i) {
        if (i != KERNEL_WIDTH) os << ", ";
        os << setw(2) << obj[i];
    }
    os << " | ";
    for (int i = KERNEL_WIDTH + BLOCK_SIZE; i < BLOCK_SIZE * 2 + KERNEL_WIDTH; ++i) {
        if (i != KERNEL_WIDTH + BLOCK_SIZE) os << ", ";
        os << setw(2) << obj[i];
    }
    os << " ]";
    return os;
}

PipedPixelsArray sum_buffered_block_0(BufferedPipedPixelsArray* buffer) {
    // Now we can calculate the sums for block 0
    PipedPixelsArray sum{};
#pragma unroll
    for (int center = 0; center < BLOCK_SIZE; ++center) {
#pragma unroll
        for (int i = -KERNEL_WIDTH; i <= KERNEL_WIDTH; ++i) {
            sum[center] += (*buffer)[KERNEL_WIDTH + center + i];
        }
    }
    return sum;
}

auto run_producer(sycl::queue& Q, sycl::host_ptr<uint16_t> image_data) -> sycl::event {
    return Q.submit([&](handler& h) {
        h.single_task<class Producer>([=]() {
            // For now, send every pixel into one pipe
            // We are using blocks based on the pipe width - this is
            // likely not an exact divisor of the fast width, so for
            // now just ignore the excess pixels
            for (size_t y = 0; y < SLOW; ++y) {
                for (size_t block = 0; block < FULL_BLOCKS; ++block) {
                    auto image_data_h = host_ptr<PipedPixelsArray>(
                      reinterpret_cast<PipedPixelsArray*>(image_data.get() + y * FAST));
                    ProducerPipeToModule::write(image_data_h[block]);
                }
            }
        });
    });
}

class KernelAccumulator {
  public:
    KernelAccumulator()
        : processing_blocks(
          reinterpret_cast<PipedPixelsArray*>(&processing_buffer[KERNEL_WIDTH]),
          2) {
        // Clear out our row storage
        for (int zr = 0; zr < FULL_KERNEL_HEIGHT; ++zr) {
            for (int zb = 0; zb < FULL_BLOCKS; ++zb) {
#pragma unroll
                for (int zp = 0; zp < BLOCK_SIZE; ++zp) {
                    rowData[zr][zb][zp] = 0;
                }
            }
        }
    }

    void start_row(PipedPixelsArray block, size_t _row) {
        processing_buffer = {};
        processing_blocks[0] = block;
        row = _row;
    }

    auto feed(size_t block_number, PipedPixelsArray block) -> PipedPixelsArray {
        processing_blocks[1] = block;
        PipedPixelsArray sum = sum_buffered_block_0(&processing_buffer);

        // Now shift everything in the row buffer to the left
        // to make room for the next pipe read
#pragma unroll
        for (int i = 0; i < KERNEL_WIDTH + BLOCK_SIZE; ++i) {
            processing_buffer[i] = processing_buffer[BLOCK_SIZE + i];
        }

        // Calculate the previously written row index, and get the row
        int prev_row_index = (row + FULL_KERNEL_HEIGHT - 1) % FULL_KERNEL_HEIGHT;
        auto prev_row = rowData[prev_row_index][block_number];
        // And the oldest row index and row (which we will replace)
        int oldest_row_index = row % FULL_KERNEL_HEIGHT;
        auto oldest_row = rowData[oldest_row_index][block_number];

        // Write the new running total over the oldest data
        PipedPixelsArray new_row = sum + prev_row;

        rowData[oldest_row_index][block_number] = new_row;

        // Now, calculate the kernel sum for each of these
        auto kernel_sum = new_row - oldest_row;
        return kernel_sum;
    }

  private:
    size_t row;
    // Storage for every kernel row, for the full detector width
    ModuleRowStore<FULL_BLOCKS> rowData;
    // The short, current local area on a row we are processing
    BufferedPipedPixelsArray processing_buffer;
    /// View onto the processing buffer
    tcb::span<PipedPixelsArray> processing_blocks;
};

auto run_module(sycl::queue& Q,
                device_ptr<uint8_t> mask_data,
                host_ptr<uint16_t> destination_data,
                host_ptr<uint16_t> destination_data_sq) -> sycl::event {
    return Q.submit([&](handler& h) {
        h.single_task<class Module>([=]() {
            auto destination_data_h = host_ptr<uint16_t>(destination_data);
            auto destination_data_sq_h = host_ptr<uint16_t>(destination_data_sq);

            KernelAccumulator sum;
            KernelAccumulator sumsq;

            for (size_t y = 0; y < SLOW; ++y) {
                auto pixels = ProducerPipeToModule::read();
                sum.start_row(pixels, y);
                sumsq.start_row(pow2(pixels), y);

                for (size_t block = 0; block < FULL_BLOCKS - 1; ++block) {
                    // Read the next block into the right of the array
                    auto pixels = ProducerPipeToModule::read();
                    auto kernel_sum = sum.feed(block, pixels);
                    auto kernel_sum_sq = sumsq.feed(block, pow2(pixels));
                    // Write this into the output data block
                    if (y >= KERNEL_HEIGHT) {
                        // Write a really simple loop.
                        size_t offset = (y - KERNEL_HEIGHT) * FAST + block * BLOCK_SIZE;
#pragma unroll
                        for (size_t i = 0; i < BLOCK_SIZE; ++i) {
                            destination_data_h[offset + i] = kernel_sum[i];
                            destination_data_sq_h[offset + i] = kernel_sum_sq[i];
                        }
                    }
                }
            }
        });
    });
}