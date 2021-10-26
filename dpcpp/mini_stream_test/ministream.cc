#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#include <array>
#include <cassert>
#include <cstdio>

using namespace sycl;

constexpr auto R = "\033[31m";
constexpr auto G = "\033[32m";
constexpr auto BOLD = "\033[1m";
constexpr auto NC = "\033[0m";

// The size of the underlying image we are trying to read
// slow = height, fast = width
constexpr size_t E2XE_16M_SLOW = 4362;
constexpr size_t E2XE_16M_FAST = 4148;

// This is a single block of pixels read every tick
// 32 = 512 bit
// 16 = 256 bit
using PipedPixelsArray = std::array<uint16_t, 8>;

class Producer;

// Convenience method to read duration ms out of an event
double event_ms(const sycl::event& e) {
    return 1e-6
           * (e.get_profiling_info<info::event_profiling::command_end>()
              - e.get_profiling_info<info::event_profiling::command_start>());
}

// Convenience method to convert a value in bytes and ms to GBps
double GBps(size_t bytes, double ms) {
    return (static_cast<double>(bytes) / 1e9) / (ms / 1000.0);
}

/// Convenience method to calculate GBps from an event and bytes transferred
double event_GBps(const sycl::event& e, size_t bytes) {
    const double ms = event_ms(e);
    return GBps(bytes, ms);
}

void fill_image_random(uint16_t* image, size_t num_pixels, int seed) {
    // An extremely simple, stable, pseudo'random' generator, Numerical Methods'
    // ranqd1 - this ensures that we have stable cross-platform results.
    // This is about as truly random as returning "nine" every time.
    int64_t idum = seed;
    for (size_t px = 0; px < num_pixels; ++px) {
        image[px] = idum % 5;
        // permute the "random" state
        idum = 1664525L * idum + 1013904223L;
    }
}

int main(int argc, char** argv) {
// Select either:
//  - the FPGA emulator device (CPU emulation of the FPGA)
//  - the FPGA device (a real FPGA)
#if defined(FPGA_EMULATOR)
    INTEL::fpga_emulator_selector device_selector;
#else
    INTEL::fpga_selector device_selector;
#endif
    queue Q(device_selector, property::queue::enable_profiling{});
    printf("Using Device: %s%s%s\n",
           BOLD,
           Q.get_device().get_info<info::device::name>().c_str(),
           NC);

    constexpr size_t NUM_PIXELS = E2XE_16M_FAST * E2XE_16M_SLOW;

    // Host-stored image data
    uint16_t* image_data = malloc_host<uint16_t>(NUM_PIXELS, Q);
    // Output array for sending data back to main memory
    size_t* result = malloc_host<size_t>(2, Q);
    // Make sure we're allocated on a 512 bit alignment
    assert(image_data % 64 == 0);

    constexpr size_t BLOCK_SIZE = std::tuple_size<PipedPixelsArray>::value;
    constexpr size_t TOTAL_BLOCKS_UNALIGNED = NUM_PIXELS / BLOCK_SIZE;

    printf("Running with %s%lu-bit%s wide blocks\n", BOLD, BLOCK_SIZE * 16, NC);
    printf("Starting image loop:\n");

    for (int i_round = 0; i_round < 8; ++i_round) {
        printf("Reading Image %d\n", i_round);
        fill_image_random(image_data, NUM_PIXELS, i_round);

        printf("Calculating host sum\n");
        // Now we are using blocks and discarding excess, do that here
        size_t host_sum = 0;
        size_t active_pixels = 0;

        constexpr size_t REPEATS = 100;
        for (int i = 0; i < TOTAL_BLOCKS_UNALIGNED * BLOCK_SIZE; ++i) {
            host_sum += image_data[i] * REPEATS;
            active_pixels += REPEATS;
        }
        printf("Starting Kernel\n");

        event e_producer = Q.submit([&](handler& h) {
            h.single_task<class Producer>([=]() {
                auto hp = host_ptr<PipedPixelsArray>(
                  reinterpret_cast<PipedPixelsArray*>(image_data));
                auto result_p = host_ptr<size_t>(result);

                size_t global_sum = 0;
                size_t num_pixels = 0;

                // We repeat to ensure that we spend lots of time in the
                // kernel to eliminate kernel-initiation-overhead
                for (size_t repeat = 0; repeat < REPEATS; ++repeat) {
                    // Probably overkill to have local variables at every
                    // level but let's be defensive
                    size_t mid_sum = 0;
                    size_t mid_num = 0;
                    for (size_t block = 0; block < TOTAL_BLOCKS_UNALIGNED; ++block) {
                        PipedPixelsArray data = hp[block];

                        size_t local_sum = 0;
                        size_t num_local = 0;
#pragma unroll
                        for (int i = 0; i < BLOCK_SIZE; ++i) {
                            local_sum += data[i];
                            num_local += 1;
                        }
                        mid_sum += local_sum;
                        mid_num += num_local;
                    }
                    global_sum += mid_sum;
                    num_pixels += mid_num;
                }
                result_p[0] = global_sum;
                result_p[1] = num_pixels;
            });
        });

        Q.wait();

        printf(" ... produced in %.2f ms (%.3g GBps)\n",
               event_ms(e_producer),
               event_GBps(e_producer, active_pixels * sizeof(uint16_t)));

        auto device_sum = result[0];
        auto color = host_sum == device_sum ? G : R;
        printf("%s     Sum = %zu / %zu%s\n", color, device_sum, host_sum, NC);

        auto color_px = result[1] == active_pixels ? G : R;
        printf("%s      px = %zu / %zu%s\n", color_px, result[1], active_pixels, NC);
    }

    free(result, Q);
    free(image_data, Q);
}
