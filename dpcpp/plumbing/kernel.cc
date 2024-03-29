#include "kernel.hpp"

#include <cmath>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "common.hpp"

using namespace sycl;

class ToModulePipe;

using ProducerPipeToModule = SYCL_INTEL::pipe<class ToModulePipe, PipedPixelsArray, 5>;

class PixelPipe;
// Pixel buffer pipe, so that we can use the same pixel later in the process
// We read KERNEL_HEIGHT full rows of data after a pixel before we have
// the complete kernel are. We need +1 block because we need to read the
// next block of pixels to account for pixels at the edge of the current
// block whose kernel area overlaps with the next block.
using PixelBufferPipe =
  SYCL_INTEL::pipe<class PixelPipe, PipedPixelsArray, KERNEL_HEIGHT * FULL_BLOCKS + 1>;

template <int Index>
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
// template<typename T = PipedPixelsArray::value_type>
using BufferedPipedPixelsArray = uint32_t[BLOCK_SIZE * 2 + KERNEL_WIDTH];

// This two-block solution only works if kernel width < block size
static_assert(KERNEL_WIDTH < BLOCK_SIZE);

// template <int blocks>
using ModuleRowStoreType = uint32_t;
using ModuleRowStore =
  std::array<ModuleRowStoreType, BLOCK_SIZE>[FULL_KERNEL_HEIGHT][FULL_BLOCKS];

template <typename T, typename U>
inline auto operator+(const std::array<T, BLOCK_SIZE>& l,
                      const std::array<U, BLOCK_SIZE>& r) {
    std::array<typename std::common_type<T, U>::type, BLOCK_SIZE> sum;
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        sum[i] = l[i] + r[i];
    }
    return sum;
}
template <typename T, typename U>
inline auto operator-(const std::array<T, BLOCK_SIZE>& l,
                      const std::array<U, BLOCK_SIZE>& r) {
    std::array<typename std::common_type<T, U>::type, BLOCK_SIZE> sub;
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        sub[i] = l[i] - r[i];
    }
    return sub;
}

template <typename T>
inline auto operator/(const std::array<T, BLOCK_SIZE>& l, float r) {
    std::array<float, BLOCK_SIZE> out;
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        out[i] = static_cast<float>(l[i]) / r;
    }
    return out;
}
template <typename T, typename U>
inline auto operator/(const std::array<T, BLOCK_SIZE>& l,
                      const std::array<U, BLOCK_SIZE>& r) {
    std::array<float, BLOCK_SIZE> out;
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        out[i] = static_cast<float>(l[i]) / static_cast<float>(r[i]);
    }
    return out;
}

template <typename Tl, typename Tr>
inline auto operator*(const Tl l, const std::array<Tr, BLOCK_SIZE>& r) {
    std::array<Tr, BLOCK_SIZE> mult;
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        mult[i] = l * r[i];
    }
    return mult;
}
template <typename Tl, typename Tr>
inline auto operator*(const std::array<Tl, BLOCK_SIZE>& l, const Tr r) {
    std::array<float, BLOCK_SIZE> mult;
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        mult[i] = l[i] * r;
    }
    return mult;
}

inline auto operator/(const PipedPixelsArray& l, std::size_t r) {
    // const std::size_t l,
    std::array<float, BLOCK_SIZE> out;
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        out[i] = static_cast<float>(l[i]) / static_cast<float>(r);
    }
    return out;
}

inline auto operator>(const std::array<float, BLOCK_SIZE>& l, float r) {
    std::array<bool, BLOCK_SIZE> out;
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        out[i] = l[i] > r;
    }
    return out;
}
inline auto operator>(const PipedPixelsArray& l,
                      const std::array<float, BLOCK_SIZE>& r) {
    std::array<bool, BLOCK_SIZE> out;
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        out[i] = l[i] > r[i];
    }
    return out;
}
inline auto sqrt(const std::array<float, BLOCK_SIZE>& in) {
    std::array<float, BLOCK_SIZE> out;
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        out[i] = std::sqrt(in[i]);
    }
    return out;
}

inline auto operator&&(const std::array<bool, BLOCK_SIZE>& l,
                       const std::array<bool, BLOCK_SIZE>& r) {
    std::array<bool, BLOCK_SIZE> out;
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        out[i] = l[i] && r[i];
    }
    return out;
}

template <typename T>
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

auto sum_buffered_block_0(BufferedPipedPixelsArray* buffer) {
    // Now we can calculate the sums for block 0
    std::array<ModuleRowStoreType, BLOCK_SIZE> sum{};
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
                    // Send every pixel into a second, delay buffer pipe
                    PixelBufferPipe::write(image_data_h[block]);
                }
            }
        });
    });
}

void initialise_row_store(ModuleRowStore& rows) {
    // Initialise this to zeros
    for (int zr = 0; zr < FULL_KERNEL_HEIGHT; ++zr) {
        for (int zb = 0; zb < FULL_BLOCKS; ++zb) {
#pragma unroll
            for (int zp = 0; zp < BLOCK_SIZE; ++zp) {
                rows[zr][zb][zp] = 0;
            }
        }
    }
}

// template <typename T>
// auto pow2(const std::array<T, BLOCK_SIZE>& val) {
//     std::array<T, BLOCK_SIZE> sqr;
// #pragma unroll
//     for (int i = 0; i < BLOCK_SIZE; ++i) {
//         sqr[i] = val[i] * val[i];
//     }
//     return sqr;
// }

auto pow2(const std::array<uint16_t, BLOCK_SIZE>& val) {
    std::array<uint32_t, BLOCK_SIZE> sqr;
#pragma unroll
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        sqr[i] = static_cast<uint32_t>(val[i]) * static_cast<uint32_t>(val[i]);
    }
    return sqr;
}

template <typename U>
auto calculate_next_block(std::size_t y,
                          std::size_t block_number,
                          ModuleRowStore& rows,
                          BufferedPipedPixelsArray& interim_pixels,
                          std::array<U, BLOCK_SIZE> new_block)
  -> std::array<ModuleRowStoreType, BLOCK_SIZE> {
    // Recreate the "block view" of this buffer
    auto* interim_blocks =
      reinterpret_cast<std::array<ModuleRowStoreType, BLOCK_SIZE>*>(
        &interim_pixels[KERNEL_WIDTH]);
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        interim_blocks[1][i] = new_block[i];
    }

    // Now we can calculate the sums for block 0
    auto sum = sum_buffered_block_0(&interim_pixels);

    // Now shift everything in the row buffer to the left
    // to make room for the next pipe read
#pragma unroll
    for (int i = 0; i < KERNEL_WIDTH + BLOCK_SIZE; ++i) {
        interim_pixels[i] = interim_pixels[BLOCK_SIZE + i];
    }
    // Calculate the previously written row index, and get the row
    int prev_row_index = (y + FULL_KERNEL_HEIGHT - 1) % FULL_KERNEL_HEIGHT;
    auto prev_row = rows[prev_row_index][block_number];
    // And the oldest row index and row (which we will replace)
    int oldest_row_index = y % FULL_KERNEL_HEIGHT;
    auto oldest_row = rows[oldest_row_index][block_number];

    // Write the new running total over the oldest data
    auto new_row = sum + prev_row;

    rows[oldest_row_index][block_number] = new_row;

    // Now, calculate the kernel sum for each of these
    auto kernel_sum = new_row - oldest_row;

    return kernel_sum;
}

auto run_module(sycl::queue& Q,
                device_ptr<uint8_t> mask_data,
                host_ptr<bool> strong_pixels,

                FindSpotsDebugOutput& debug_data) -> sycl::event {
    return Q.submit([&](handler& h) {
        h.single_task<class Module<0>>([=]() {
#ifdef DEBUG_IMAGES
            auto debug = const_cast<FindSpotsDebugOutput&>(debug_data);
#endif

            auto strong_pixels_h = host_ptr<bool>(strong_pixels);
            size_t sum_pixels = 0;
            size_t strong_pixels_count = 0;

            // Make a buffer for full rows so we can store them as we go
            ModuleRowStore rows, rows_sq;

            initialise_row_store(rows);
            initialise_row_store(rows_sq);

            for (size_t y = 0; y < SLOW; ++y) {
                // The per-pixel buffer array to accumulate the blocks
                BufferedPipedPixelsArray interim_pixels{};
                BufferedPipedPixelsArray interim_pixels_sq{};

                auto pixels = ProducerPipeToModule::read();

                // Read the first block into initial position in the array
                auto* interim_blocks =
                  reinterpret_cast<PipedPixelsArray*>(&interim_pixels[KERNEL_WIDTH]);
                auto* interim_blocks_sq =
                  reinterpret_cast<std::array<uint32_t, BLOCK_SIZE>*>(
                    &interim_pixels_sq[KERNEL_WIDTH]);
                interim_blocks[0] = pixels;
                interim_blocks_sq[0] = pow2(pixels);

                for (size_t block = 0; block < FULL_BLOCKS - 1; ++block) {
                    auto pixels = ProducerPipeToModule::read();

                    auto kernel_sum =
                      calculate_next_block(y, block, rows, interim_pixels, pixels);
                    auto kernel_sum_sq = calculate_next_block(
                      y, block, rows_sq, interim_pixels_sq, pow2(pixels));

                    // Until we reach the kernel height, we aren't ready to calculate anything
                    if (y < KERNEL_HEIGHT) {
                        continue;
                    }

                    // Get the value of the pixels KERNEL_HEIGHT rows ago
                    auto kernel_px = PixelBufferPipe::read();

                    // Calculate the thresholding values for these kernels.
                    // Let's assume for now that everything is unmasked.
                    constexpr float N =
                      (KERNEL_WIDTH * 2 + 1) * (KERNEL_HEIGHT * 2 + 1);
                    auto background_threshold =
                      1 + sigma_background * std::sqrt(2 / (N - 1));

                    size_t _count = 0;
                    // Let's write back to our host for introspection
                    size_t offset = (y - KERNEL_HEIGHT) * FAST + block * BLOCK_SIZE;
#pragma unroll
                    for (size_t i = 0; i < BLOCK_SIZE; ++i) {
                        float sum = kernel_sum[i];
                        float sum_sq = kernel_sum_sq[i];

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

// #pragma unroll
//                         for (size_t i = 0; i < BLOCK_SIZE; ++i) {
#ifdef DEBUG_IMAGES
                        if (variance < 0) variance = -1;
                        if (dispersion < 0) dispersion = -1;

                        debug.sum[offset + i] = sum;
                        debug.sumsq[offset + i] = sum_sq;
                        debug.dispersion[offset + i] = dispersion;
                        debug.mean[offset + i] = mean;
                        debug.variance[offset + i] = variance;
#endif
                        strong_pixels_h[offset + i] = is_strong_pixel;
                    }
                    strong_pixels_count += _count;
                }  // blocks
                // We ignore the last block in the algorithm
                if (y >= KERNEL_HEIGHT) {
                    PixelBufferPipe::read();
                }
            }  // y

            // Drain the pipe - because we don't do the bottom edge of the image,
            // we have KERNEL_HEIGHT rows of pixel data that has remained unread
            for (int i = 0; i < FULL_BLOCKS * KERNEL_HEIGHT; ++i) {
                PixelBufferPipe::read();
            }
        });
    });
}