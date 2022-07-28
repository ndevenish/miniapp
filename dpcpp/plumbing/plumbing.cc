
#include <fmt/core.h>
#include <inttypes.h>

#include <CL/sycl.hpp>
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdlib>

#include "common.hpp"
#include "eiger2xe.h"
#include "h5read.h"
#include "kernel.hpp"

using namespace sycl;

// Width of this array determines how many pixels we read at once
// class PipedPixelsArray {
//   public:
//     typedef H5Read::image_type value_type;

//     value_type data[BLOCK_SIZE];

//     const value_type& operator[](size_t index) const {
//         return this->data[index];
//     }
//     value_type& operator[](size_t index) {
//         return this->data[index];
//     }
// };

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

/// Return the profiling event time, in milliseconds, for an event
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

/// Draw a subset of the pixel values for a 2D image array
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

void check_allocs() {}
/// Basic sanity check on allocations - so that if they fail, we don't get a SEGV later
template <typename T, typename... R>
void check_allocs(T arg, R... args) {
    if (arg == nullptr) {
        throw std::bad_alloc{};
    }
    check_allocs(args...);
}

int main(int argc, char** argv) {
    auto start_time = std::chrono::high_resolution_clock::now();

    // Parse arguments and get our H5Reader
    auto parser = FPGAArgumentParser();
    parser.add_h5read_arguments();
    auto args = parser.parse_args(argc, argv);
    auto reader = args.file.empty() ? H5Read() : H5Read(args.file);

    sycl::queue Q{args.device(), sycl::property::queue::enable_profiling{}};

    fmt::print("Running with {}{}-bit{} wide blocks\n", BOLD, BLOCK_SIZE * 16, NC);

    auto slow = reader.get_image_slow();
    auto fast = reader.get_image_fast();
    const size_t num_pixels = slow * fast;
    // Make sure these match our hardcoded values
    assert(slow == SLOW);
    assert(fast == FAST);

    // Mask data is the same for all images, so we copy it to device early
    auto mask_data = device_ptr<uint8_t>(malloc_device<uint8_t>(num_pixels, Q));
    // Declare the image data that will be remotely accessed
    auto image_data = host_ptr<uint16_t>(malloc_host<uint16_t>(num_pixels, Q));
    check_allocs(mask_data, image_data);
    // Paranoia: Ensure that this is properly aligned
    assert(reinterpret_cast<uintptr_t>(image_data.get()) % 64 == 0);

    auto row_count = host_ptr<uint16_t>(malloc_host<uint16_t>(1, Q));
    check_allocs(row_count);
    *row_count = 0;
    // Result data - this is accessed remotely but only at the end, to
    // return the results.
    uintptr_t* result_dest = malloc_host<uintptr_t>(BLOCK_SIZE + 1, Q);
    uint16_t* result_mini = malloc_host<uint16_t>(BLOCK_SIZE, Q);

    auto* result = malloc_host<PipedPixelsArray>(2, Q);

    printf("Uploading mask data to accelerator.... ");
    auto e_mask_upload = Q.submit(
      [&](handler& h) { h.memcpy(mask_data, reader.get_mask().data(), num_pixels); });
    Q.wait();
    printf("done in %.1f ms (%.2f GBps)\n",
           event_ms(e_mask_upload),
           event_GBps(e_mask_upload, num_pixels));

    printf(
      "Block data:\n"
      "         SIZE: %zu px per block\n"
      "    REMAINDER: %zu px unprocessed per row\n"
      "  FULL_BLOCKS: %zu blocks across image width\n",
      BLOCK_SIZE,
      BLOCK_REMAINDER,
      FULL_BLOCKS);

    // uint16_t* totalblocksum = malloc_host<uint16_t>(FULL_BLOCKS * slow, Q);

    // auto* destination_data_host = malloc_device<uint16_t>(num_pixels, Q);
    auto* destination_data = malloc_host<uint16_t>(num_pixels, Q);
    // Fill this with sample data so we can tell if anything is happening
    for (size_t i = 0; i < num_pixels; ++i) {
        destination_data[i] = 42;
    }
    // auto* rows_ptr = malloc_device<ModuleRowStore<FULL_BLOCKS>>(1, Q);
    // auto  rows = malloc_device<
    //                 //                        FULL_KERNEL_HEIGHT>{};
    Q.wait();
    printf("Starting image loop:\n");
    for (int i = 0; i < reader.get_number_of_images(); ++i) {
        printf("\nReading Image %d\n", i);
        reader.get_image_into(i, image_data);

        // Precalculate host-side the answers we expect, so we can validate
        printf("Calculating host sum\n");
        // Now we are using blocks and discarding excess, do that here
        size_t host_sum = 0;
        for (int i = 0; i < FULL_BLOCKS * BLOCK_SIZE; ++i) {
            host_sum += image_data[i];
        }
        printf("Starting Kernels\n");
        auto t1 = std::chrono::high_resolution_clock::now();

        event e_producer = run_producer(Q, image_data, slow, fast);
        event e_module = run_module(Q, mask_data, destination_data);
        Q.wait();

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

        printf("Data store instructs to %" PRIxPTR " ==? %" PRIxPTR "\n",
               result_dest[BLOCK_SIZE],
               (uintptr_t)destination_data);
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            printf(" %" PRIxPTR, result_dest[i]);
        }
        printf("\nData:\n");
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            printf(" %" PRIu16, result_mini[i]);
        }
        printf("\nData on host:");
        //   &destination_data_h[(y - KERNEL_HEIGHT) * fast
        //                       + block * BLOCK_SIZE]) = kernel_sum;
        size_t offset = (5 - KERNEL_HEIGHT) * fast;
        printf("%" PRIxPTR ", %" PRIxPTR "\n", offset, (uintptr_t)destination_data);
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            printf(" %" PRIu16, destination_data[offset + i]);
        }
        printf("\n");
        // Copy the device destination buffer back
        // auto host_sum_data = host_ptr<uint16_t>(malloc_host<uint16_t>(num_pixels, Q));

        // auto e_dest_download = Q.submit([&](handler& h) {
        //     h.memcpy(host_sum_data, destination_data, num_pixels * sizeof(uint16_t));
        // });
        // e_dest_download.wait();
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
    free(mask_data, Q);
    auto end_time = std::chrono::high_resolution_clock::now();

    printf(
      "Total run duration: %.2f s\n",
      std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time)
        .count());
}
