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

double Gbps(size_t bytes, double ms) {
    return (static_cast<double>(bytes) * 8.0 / 1e9) / (ms / 1000.0);
}

/// Return value of event in terms of Gigabits per second
double event_Gbps(const sycl::event& e, size_t bytes) {
    const double ms = event_ms(e);
    return Gbps(bytes, ms);
}

// From https://stackoverflow.com/a/66146159/1118662
constexpr int int_ceil(float f) {
    const int i = static_cast<int>(f);
    return f > i ? i + 1 : i;
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

    const size_t num_pixels = reader.get_image_slow() * reader.get_image_fast();

    // Mask data is the same for all images, so we copy it early
    uint8_t* mask_data = malloc_device<uint8_t>(num_pixels, Q);
    uint16_t* image_data = malloc_host<uint16_t>(num_pixels, Q);
    size_t* result = malloc_shared<size_t>(2, Q);

    fmt::print("Uploading mask data to accelerator.... ");
    auto e_mask_upload = Q.submit(
      [&](handler& h) { h.memcpy(mask_data, reader.get_mask().data(), num_pixels); });
    Q.wait();
    fmt::print("done in {:.1f} ms ({:.2f} Gbps)\n",
               event_ms(e_mask_upload),
               event_Gbps(e_mask_upload, num_pixels));

    fmt::print("Starting image loop:\n");
    for (int i = 0; i < reader.get_number_of_images(); ++i) {
        fmt::print("\nReading Image {}\n", i);
        reader.get_image_into(i, image_data);

        fmt::print("Calculating host sum\n");
        size_t host_sum = 0;
        for (int px = 0; px < num_pixels; ++px) {
            host_sum += image_data[px];
        }
        fmt::print("Starting Kernels\n");
        auto t1 = std::chrono::high_resolution_clock::now();

        constexpr size_t BLOCK_SIZE = std::tuple_size<PipedPixelsArray>::value;
        constexpr size_t BLOCK_REMAINDER = E2XE_16M_FAST % BLOCK_SIZE;
        constexpr size_t FULL_BLOCKS = (E2XE_16M_FAST - BLOCK_REMAINDER) / BLOCK_SIZE;
        constexpr size_t NUM_BLOCKS = int_ceil(E2XE_16M_FAST / BLOCK_SIZE);
        fmt::print("{} blocks ({} full) of {} width, last block padded with {} zeros",
                   NUM_BLOCKS,
                   FULL_BLOCKS,
                   BLOCK_SIZE,
                   BLOCK_REMAINDER);

        event e_producer = Q.submit([&](handler& h) {
            h.single_task<class Producer>([=]() {
                // For now, send every pixel into one pipe
                // static_assert(E2XE_16M_FAST %
                // std::tuple_size<PipedPixelsArray>::value
                //               == 0);
                // size_t i = 0;
                for (size_t i = 0; i < num_pixels; i += E2XE_16M_FAST) {
                    // Now we have a "remainder" block we want to send
                    // We need to use the initial portion and pad with zeros
                    PipedPixelsArray remain{};
#pragma unroll
                    for (size_t r = 0; r < BLOCK_SIZE - BLOCK_REMAINDER; ++r) {
                        remain[i] = image_data[i + FULL_BLOCKS * BLOCK_SIZE + r];
                    }

                    // Do "real" blocks
                    for (size_t block = 0; block < FULL_BLOCKS; ++block) {
                        ProducerPipeToModule<0>::write(
                          *reinterpret_cast<PipedPixelsArray*>(image_data + i
                                                               + block * BLOCK_SIZE));
                    }
                    if constexpr (BLOCK_REMAINDER > 0) {
                        ProducerPipeToModule<0>::write(remain);
                    }
                }

                // }
                // for (size_t i = 0; i < num_pixels;)
                // for (size_t i = 0; i < num_pixels;
                //  i += std::tuple_size<PipedPixelsArray>::value) {
                //      ProducerPipeToModule<0>::write(
                //        *reinterpret_cast<PipedPixelsArray*>(image_data + i));
                //  }
            });
        });

        // Launch a module kernel for every module
        event e_module = Q.submit([&](handler& h) {
            h.single_task<class Module<0>>([=](){
                size_t sum_pixels = 0;
                for (size_t i = 0; i < NUM_BLOCKS * E2XE_16M_FAST; ++i) {
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
        // double ms = max(event_ms(e_producer), event_ms(e_producer_2));
        double ms_all =
          std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count()
          * 1000;

        fmt::print(" ... produced in {:.2f} ms ({:.3g} Gbps)\n",
                   event_ms(e_producer),
                   event_Gbps(e_producer, num_pixels * sizeof(uint16_t) / 2));
        fmt::print(" ... consumed in {:.2f} ms ({:.3g} Gbps)\n",
                   event_ms(e_module),
                   event_Gbps(e_module, num_pixels * sizeof(uint16_t) / 2));
        fmt::print(" ... Total consumed + piped in host time {:.2f} ms ({:.3g} Gbps)\n",
                   ms_all,
                   Gbps(num_pixels * sizeof(uint16_t), ms_all));

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
