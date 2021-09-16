#include <fmt/color.h>
#include <fmt/core.h>

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>

#include "eiger2xe.h"
#include "h5read.h"

constexpr auto R = "\033[31m";
constexpr auto G = "\033[32m";
constexpr auto Y = "\033[33m";
constexpr auto B = "\033[34m";
constexpr auto GRAY = "\033[37m";
constexpr auto BOLD = "\033[1m";
constexpr auto NC = "\033[0m";

using namespace sycl;

template <int id>
class ToModulePipe;

using PipedPixelsArray = std::array<H5Read::image_type, 16>;

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

int main(int argc, char** argv) {
    auto start_time = std::chrono::high_resolution_clock::now();
    auto reader = H5Read(argc, argv);

#ifdef FPGA
// Select either:
//  - the FPGA emulator device (CPU emulation of the FPGA)
//  - the FPGA device (a real FPGA)
#if defined(FPGA_EMULATOR)
    INTEL::fpga_emulator_selector device_selector;
#else
    INTEL::fpga_selector device_selector;
#endif
    queue Q(device_selector, property::queue::enable_profiling{});
#else
    queue Q{property::queue::enable_profiling{}};
#endif

    // Print information about the device we are using
    std::string device_kind = Q.get_device().is_cpu()           ? "CPU"
                              : Q.get_device().is_gpu()         ? "GPU"
                              : Q.get_device().is_accelerator() ? "FPGA"
                                                                : "Unknown";
    fmt::print("Using {0}{2}{1} Device: {0}{3}{1}\n\n",
               BOLD,
               NC,
               device_kind,
               Q.get_device().get_info<info::device::name>());

    auto slow = reader.get_image_slow();
    auto fast = reader.get_image_fast();
    const size_t num_pixels = reader.get_image_slow() * reader.get_image_fast();

    // Mask data is the same for all images, so we copy it early
    uint8_t* mask_data = malloc_device<uint8_t>(num_pixels, Q);
    uint16_t* image_data = malloc_host<uint16_t>(num_pixels, Q);
    size_t* result = malloc_shared<size_t>(2, Q);

    fmt::print("Uploading mask data to accelerator.... ");
    auto e_mask_upload = Q.submit(
      [&](handler& h) { h.memcpy(mask_data, reader.get_mask().data(), num_pixels); });
    Q.wait();
    fmt::print("done in {:.1f} ms ({:.2f} GBps)\n",
               event_ms(e_mask_upload),
               event_GBps(e_mask_upload, num_pixels));

    fmt::print("Starting image loop:\n");
    for (int i = 0; i < reader.get_number_of_images(); ++i) {
        fmt::print("\nReading Image {}\n", i);
        reader.get_image_into(i, image_data);

        constexpr size_t BLOCK_SIZE = std::tuple_size<PipedPixelsArray>::value;
        constexpr size_t BLOCK_REMAINDER = E2XE_16M_FAST % BLOCK_SIZE;
        constexpr size_t FULL_BLOCKS = (E2XE_16M_FAST - BLOCK_REMAINDER) / BLOCK_SIZE;

        fmt::print("Calculating host sum\n");
        // Now we are using blocks and discarding excess, do that here
        size_t host_sum = 0;
        for (int y = 0; y < slow; ++y) {
            for (int px = 0; px < BLOCK_SIZE * FULL_BLOCKS; ++px) {
                host_sum += image_data[y * fast + px];
            }
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
                    size_t i = y * E2XE_16M_FAST;
                    for (size_t block = 0; block < FULL_BLOCKS; ++block) {
                        ProducerPipeToModule<0>::write(
                          *reinterpret_cast<PipedPixelsArray*>(image_data + i));
                        i += BLOCK_SIZE;
                    }
                }
            });
        });

        // Launch a module kernel for every module
        event e_module = Q.submit([&](handler& h) {
            h.single_task<class Module<0>>([=](){
                size_t sum_pixels = 0;

                // We have evenly sized blocks send to us
                for (size_t block = 0; block < FULL_BLOCKS * slow; ++block) {
                    PipedPixelsArray data = ProducerPipeToModule<0>::read();
#pragma unroll
                    for (uint16_t px : data) {
                        sum_pixels += px;
                    }
                }
                result[0] = sum_pixels;
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

        auto device_sum = result[0];
        auto color = fg(host_sum == device_sum ? fmt::color::green : fmt::color::red);
        fmt::print(color, "     Sum = {} / {}\n", device_sum, host_sum);
    }

    free(result, Q);
    free(image_data, Q);
    free(mask_data, Q);
    auto end_time = std::chrono::high_resolution_clock::now();
    ;
    fmt::print(
      "Total run duration: {:.2f} s\n",
      std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time)
        .count());
}
