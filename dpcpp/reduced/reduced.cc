
#include <inttypes.h>

#include <CL/sycl.hpp>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>

// Abstract over OneAPI versions
#if __INTEL_LLVM_COMPILER < 20220000
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#define SYCL_INTEL sycl::INTEL
#else
#include <sycl/ext/intel/fpga_extensions.hpp>
#define SYCL_INTEL sycl::ext::intel
#endif

using namespace sycl;

////////////////////////////////////////////////////////////////////////
/// The data type of individual image data pixels
using image_data_t = uint16_t;

////////////////////////////////////////////////////////////////////////
// Defining geometry for the detector images

/// Number of pixels Y in the image
constexpr size_t E2XE_16M_SLOW = 4362;
/// Number of pixels X in the image
constexpr size_t E2XE_16M_FAST = 4148;
/// Number of modules Y in the image
constexpr size_t E2XE_16M_NSLOW = 8;
/// Number of modules X in the image
constexpr size_t E2XE_16M_NFAST = 4;
/// Number of X pixels in a module
constexpr size_t E2XE_MOD_FAST = 1028;
/// Number of Y pixels in a module
constexpr size_t E2XE_MOD_SLOW = 512;
/// Number of X pixels between modules
constexpr size_t E2XE_GAP_FAST = 12;
/// Number of Y pixels between modules
constexpr size_t E2XE_GAP_SLOW = 38;

////////////////////////////////////////////////////////////////////////
/// Generate a set of sample images
constexpr size_t NUM_SAMPLE_IMAGES = 4;
void generate_sample_image(size_t n, image_data_t* data) {
    assert(n >= 0 && n <= NUM_SAMPLE_IMAGES);

    if (n == 0) {
        memset(data, 0, E2XE_16M_FAST * E2XE_16M_SLOW * sizeof(uint16_t));
    } else if (n == 1) {
        // Image 1: I=1 for every unmasked pixel
        memset(data, 0, E2XE_16M_FAST * E2XE_16M_SLOW * sizeof(uint16_t));
        for (int mody = 0; mody < E2XE_16M_NSLOW; ++mody) {
            // row0 is the row of the module top row
            size_t row0 = mody * (E2XE_MOD_SLOW + E2XE_GAP_SLOW);
            for (int modx = 0; modx < E2XE_16M_NFAST; ++modx) {
                // col0 is the column of the module left
                int col0 = modx * (E2XE_MOD_FAST + E2XE_GAP_FAST);
                for (int row = 0; row < E2XE_MOD_SLOW; ++row) {
                    for (int x = 0; x < E2XE_MOD_FAST; ++x) {
                        *(data + E2XE_16M_FAST * (row0 + row) + col0 + x) = 1;
                    }
                }
            }
        }
    } else if (n == 2) {
        // Image 2: High pixel (100) every 42 pixels across the detector
        memset(data, 0, E2XE_16M_FAST * E2XE_16M_SLOW * sizeof(uint16_t));
        for (int y = 0; y < E2XE_16M_SLOW; y += 42) {
            for (int x = 0; x < E2XE_16M_FAST; x += 42) {
                int k = y * E2XE_16M_FAST + x;
                data[k] = 100;
            }
        }
    } else if (n == 3) {
        // Image 3: "Random" background, zero on masks

        // Implement a very simple 'random' generator, Numerical Methods' ranqd1
        // - this ensures that we have stable cross-platform results.
        uint64_t idum = 0;

        memset(data, 0, E2XE_16M_FAST * E2XE_16M_SLOW * sizeof(uint16_t));
        for (int mody = 0; mody < E2XE_16M_NSLOW; ++mody) {
            // row0 is the row of the module top row
            size_t row0 = mody * (E2XE_MOD_SLOW + E2XE_GAP_SLOW);
            for (int modx = 0; modx < E2XE_16M_NFAST; ++modx) {
                // col0 is the column of the module left
                int col0 = modx * (E2XE_MOD_FAST + E2XE_GAP_FAST);
                for (int row = 0; row < E2XE_MOD_SLOW; ++row) {
                    for (int x = 0; x < E2XE_MOD_FAST; ++x) {
                        *(data + E2XE_16M_FAST * (row0 + row) + col0 + x) = (idum % 10);
                        // basic LCG % 4 isn't unpredictable enough for us. Fake it.
                        do {
                            idum = 1664525UL * idum + 1013904223UL;
                        } while (idum % 10 >= 4);
                    }
                }
            }
        }
    } else {
        fprintf(stderr, "Error: Unhandled sample image %d\n", (int)n);
        exit(2);
    }
}

// For static asserting the block layouts
// From https://stackoverflow.com/a/10585543/1118662
constexpr bool is_power_of_two(unsigned int x) {
    return x != 0 && ((x & (x - 1)) == 0);
}

/// One-direction (half-)width of kernel. Total kernel span is (K_W * 2 + 1)
constexpr int KERNEL_WIDTH = 3;
/// One-direction (half-)height of kernel. Total kernel span is (K_H * 2 + 1)
constexpr int KERNEL_HEIGHT = 3;
constexpr int FULL_KERNEL_HEIGHT = KERNEL_HEIGHT * 2 + 1;

// How many pixels we use at once
constexpr size_t BLOCK_SIZE = 16;
static_assert(is_power_of_two(BLOCK_SIZE));

// Width of this array determines how many pixels we read at once.
// This used to be std::array to have block assignments, but was
// advised to use a struct instead.
class PipedPixelsArray {
  public:
    using value_type = image_data_t;

    value_type data[BLOCK_SIZE];

    const value_type& operator[](size_t index) const {
        return this->data[index];
    }
    value_type& operator[](size_t index) {
        return this->data[index];
    }
};

// Summing two blocks of data together
auto operator+(const PipedPixelsArray& l, const PipedPixelsArray& r)
  -> PipedPixelsArray {
    PipedPixelsArray sum;
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        sum.data[i] = l.data[i] + r.data[i];
    }
    return sum;
}
// Subtracting two blocks of data
auto operator-(const PipedPixelsArray& l, const PipedPixelsArray& r)
  -> PipedPixelsArray {
    PipedPixelsArray sum;
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        sum.data[i] = l.data[i] - r.data[i];
    }
    return sum;
}
// Convenience; printing a block of data
const sycl::stream& operator<<(const sycl::stream& os, const PipedPixelsArray& obj) {
    os << "[ ";
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        if (i > 0) {
            os << ", ";
        }
        os << setw(2) << obj[i];
    }
    os << " ]";
    return os;
}

// Algorithm is: for every pixel in a row, calculate the (self-inclusive) sum
// of all pixels within KERNEL_WIDTH of that pixel to the left and right.
//
// Thus, when scanning along the rows we need to buffer two blocks + kernel,
// because the pixels on the beginning of the block depend on the tail of the
// previous block, and the pixels at the end of the block depend on the start
// of the next block.
//
// Let's make a rolling buffer of:
//
//      | <KERNEL_WIDTH> | Block 0 | Block 1 |
//
// We read a block into block 1 - at which point we are ready to calculate all
// of the local-kernel sums for every pixel in block 0 e.g.:
//
//      | K-2 | K-1 | k-0 | B0_0 | B0_1 | B0_2 | B0_3
//         └─────┴─────┴──────┼──────┴──────┴─────┘
//                            +
//                            │
//                         | S_0 | S_1 | S_2 | S_3 | ...
//
// Once we've calculated the per-pixel kernel sum for a single block, we can
// shift the entire array left by BLOCK_SIZE + KERNEL_WIDTH pixels to read the
// next block into the right of the buffer.
//
// Since we only need the raw pixel values of the buffer+block, this process
// can be pipelined.
using BufferedPipedPixelsArray =
  PipedPixelsArray::value_type[BLOCK_SIZE * 2 + KERNEL_WIDTH];
// This two-block solution only works if kernel width < block size
static_assert(KERNEL_WIDTH < BLOCK_SIZE);

/// Calculate the kernel-width sum for every pixel in block 0
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

// Convenience method for printing out a BufferedPipedPixelsArray
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

/// Storing a whole row of data
template <int blocks>
using ModuleRowStore = PipedPixelsArray[FULL_KERNEL_HEIGHT][blocks];

/// Pipe between modules
template <int id>
class ToModulePipe;

template <int id>
using ProducerPipeToModule =
  SYCL_INTEL::pipe<class ToModulePipe<id>, PipedPixelsArray, 5>;

template <int Index>
class Module;

class Producer;

/// Convenience: Return the profiling event time, in milliseconds, for an event
double event_ms(const sycl::event& e) {
    return 1e-6
           * static_cast<double>(
             e.get_profiling_info<info::event_profiling::command_end>()
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

/// Ascii-draw a subset of the pixel values for a 2D image array
/// fast, slow, width, height - describe the bounding box to draw
/// data_width, data_height - describe the full data array size
void draw_image_data(const uint16_t* data,
                     size_t fast,
                     size_t slow,
                     size_t width,
                     size_t height,
                     size_t data_width,
                     size_t data_height) {
    for (int y = slow; y < min(slow + height, data_height); ++y) {
        if (y == slow) {
            printf("y = %2d │", y);
        } else {
            printf("    %2d │", y);
        }
        for (int i = fast; i < fast + width; ++i) {
            printf("%3d  ", data[i + data_width * y]);
        }
        printf("│\n");
    }
}

int main(int argc, char** argv) {
    auto start_time = std::chrono::high_resolution_clock::now();

#if defined(FPGA_EMULATOR)
    SYCL_INTEL::fpga_emulator_selector device_selector;
#else
    SYCL_INTEL::fpga_selector device_selector;
#endif
    sycl::queue Q(device_selector, sycl::property::queue::enable_profiling{});

    printf("Running with \033[1m%zu-bit\033[0m wide blocks\n", BLOCK_SIZE * 16);

    auto slow = E2XE_16M_SLOW;
    auto fast = E2XE_16M_SLOW;
    const size_t num_pixels = slow * fast;

    // Declare the image data that will be remotely accessed
    auto image_data = host_ptr<uint16_t>(malloc_host<uint16_t>(num_pixels, Q));
    // Absolutely make sure that this is properly aligned
    assert(reinterpret_cast<uintptr_t>(image_data.get()) % 64 == 0);

    auto row_count = host_ptr<uint16_t>(malloc_host<uint16_t>(1, Q));
    *row_count = 0;
    // Result data - this is accessed remotely but only at the end, to
    // stream the results back to the host.
    uintptr_t* result_dest = malloc_host<uintptr_t>(BLOCK_SIZE + 1, Q);
    uint16_t* result_mini = malloc_host<uint16_t>(BLOCK_SIZE, Q);
    auto* result = malloc_host<PipedPixelsArray>(2, Q);

    // Module/detector compile-time calculations
    // A full image width isn't exactly a multiple of BLOCK_SIZE.
    // Ignore this for now, and don't calculate for the pixels on the right edge.
    //
    /// The number of pixels left over when we divide the image into blocks of BLOCK_SIZE
    constexpr size_t BLOCK_REMAINDER = E2XE_16M_FAST % BLOCK_SIZE;
    // The number of full blocks that we can fit across an image
    constexpr size_t FULL_BLOCKS = (E2XE_16M_FAST - BLOCK_REMAINDER) / BLOCK_SIZE;

    printf(
      "Block data:\n"
      "         SIZE: %zu px per block\n"
      "    REMAINDER: %zu px unprocessed per row\n"
      "  FULL_BLOCKS: %zu blocks across image width\n",
      BLOCK_SIZE,
      BLOCK_REMAINDER,
      FULL_BLOCKS);

    // Make both host and device buffers - let's try explicit buffer transfers
    auto* destination_data = malloc_host<uint16_t>(num_pixels, Q);
    auto* destination_data_device = malloc_device<uint16_t>(num_pixels, Q);
    Q.wait();
    // Fill this with sample data so we can tell if it's actually overwritten
    for (size_t i = 0; i < num_pixels; ++i) {
        destination_data[i] = 42;
    }
    // Copy to accelerator
    auto e_data_clear_upload = Q.submit([&](handler& h) {
        h.memcpy(
          destination_data_device, destination_data, num_pixels * sizeof(uint16_t));
    });
    Q.wait();
    printf("Copy of empty destination done in %.1f ms (%.2f GBps)\n",
           event_ms(e_data_clear_upload),
           event_GBps(e_data_clear_upload, num_pixels));

    printf("Starting image loop:\n");
    for (int i = 0; i < NUM_SAMPLE_IMAGES; ++i) {
        printf("\nReading Image %d\n", i);
        generate_sample_image(i, image_data);

        printf("Starting Kernels\n");
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
            h.single_task<class Module<0>>([=]() {
                auto destination_data_d = device_ptr<uint16_t>(destination_data_device);

                size_t sum_pixels = 0;

                // Make a buffer for full rows to process intermediate results
                ModuleRowStore<FULL_BLOCKS> rows;
                // Initialise this to zeros
                for (int zr = 0; zr < FULL_KERNEL_HEIGHT; ++zr) {
                    for (int zb = 0; zb < FULL_BLOCKS; ++zb) {
#pragma unroll
                        for (int zp = 0; zp < BLOCK_SIZE; ++zp) {
                            rows[zr][zb][zp] = 0;
                        }
                    }
                }

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
                        PipedPixelsArray sum = sum_buffered_block_0(&interim_pixels);

                        // Now shift everything in the row buffer to the left
                        // to make room for the next pipe read
#pragma unroll
                        for (int i = 0; i < KERNEL_WIDTH + BLOCK_SIZE; ++i) {
                            interim_pixels[i] = interim_pixels[BLOCK_SIZE + i];
                        }

                        // Now we can insert this into the row accumulation store and
                        // do per-row calculations

                        // Calculate the previously written row index, and read the block
                        int prev_row_index =
                          (y + FULL_KERNEL_HEIGHT - 1) % FULL_KERNEL_HEIGHT;
                        auto prev_row = rows[prev_row_index][block];
                        // And the oldest row index and row (which we will replace)
                        int oldest_row_index = y % FULL_KERNEL_HEIGHT;
                        auto oldest_row = rows[oldest_row_index][block];

                        // Write the new running total over the oldest data
                        PipedPixelsArray new_row = sum + prev_row;
                        rows[oldest_row_index][block] = new_row;

                        // Now, calculate the kernel sum for each of these
                        auto kernel_sum = new_row - oldest_row;

                        // Write this into the output data block
                        if (y >= KERNEL_HEIGHT) {
                            // Write a really simple loop. to see if this is the problem
                            size_t offset =
                              (y - KERNEL_HEIGHT) * fast + block * BLOCK_SIZE;
#pragma unroll
                            for (size_t i = 0; i < BLOCK_SIZE; ++i) {
                                destination_data_d[offset + i] = kernel_sum[i];
                            }
                            // Previous approach was as an assignment;
                            // *reinterpret_cast<PipedPixelsArray*>(
                            //   &destination_data_h[(y - KERNEL_HEIGHT) * fast
                            //                       + block * BLOCK_SIZE]) = kernel_sum;
                        }
                    }
                }
            });
        });

        Q.wait();

        // Print out a load of diagnostics about how fast this was
        auto t2 = std::chrono::high_resolution_clock::now();
        double ms_all =
          std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count()
          * 1000;
        printf(" ... produced in %.2f ms (%.3g GBps)\n",
               event_ms(e_producer),
               event_GBps(e_producer, num_pixels * sizeof(uint16_t) / 2));
        printf(" ... consumed in %.2f ms (%.3g GBps)\n",
               event_ms(e_module),
               event_GBps(e_module, num_pixels * sizeof(uint16_t) / 2));
        printf(" ... Total consumed + piped in host time %.2f ms (%.3g GBps)\n",
               ms_all,
               GBps(num_pixels * sizeof(uint16_t), ms_all));

        // Copy the device destination buffer back
        auto e_processed_data_download = Q.submit([&](handler& h) {
            h.memcpy(
              destination_data, destination_data_device, num_pixels * sizeof(uint16_t));
        });
        e_processed_data_download.wait();
        printf("Copy back of processed data in %.1f ms (%.2f GBps)\n",
               event_ms(e_processed_data_download),
               event_GBps(e_processed_data_download, num_pixels));

        Q.wait();

        // Print a section of the image and "destination" arrays
        printf("Data:\n");
        draw_image_data(image_data.get(), 0, 0, 16, 16, fast, slow);

        printf("\nSum:\n");
        draw_image_data(destination_data, 0, 0, 16, 16, fast, slow);
        // free(host_sum_data, Q);
    }

    free(result, Q);
    free(result_dest, Q);
    free(image_data, Q);
    free(destination_data_device, Q);
    auto end_time = std::chrono::high_resolution_clock::now();

    printf(
      "Total run duration: %.2f s\n",
      std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time)
        .count());
}
