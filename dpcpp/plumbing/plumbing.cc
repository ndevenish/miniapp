#include <fmt/color.h>
#include <fmt/core.h>

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#include <algorithm>
#include <iostream>

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

template <int id>
using ProducerPipeToModule = INTEL::pipe<class ToModulePipe<id>, H5Read::image_type, 5>;

template <int Index>
class Module;

class Producer;

double event_ms(const sycl::event& e) {
    return 1e-6
           * (e.get_profiling_info<info::event_profiling::command_end>()
              - e.get_profiling_info<info::event_profiling::command_start>());
}

int main(int argc, char** argv) {
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
    size_t* result = malloc_shared<size_t>(1, Q);
    fmt::print("Uploading mask data to accelerator\n");
    auto e_mask_upload = Q.submit(
      [&](handler& h) { h.memcpy(mask_data, reader.get_mask().data(), num_pixels); });
    Q.wait();
    fmt::print(" ...done in {:.1f} ms\n", event_ms(e_mask_upload));

    fmt::print("Starting image loop:\n");
    for (int i = 0; i < reader.get_number_of_images(); ++i) {
        fmt::print("Reading Image {}\n", i);
        reader.get_image_into(i, image_data);
        size_t host_sum = 0;
        for (int px = 0; px < num_pixels; ++px) {
            host_sum += image_data[px];
        }

        event e_producer = Q.submit([&](handler& h) {
            h.single_task<class Producer>([=]() {
                // For now, send every pixel into one pipe
                for (size_t i = 0; i < num_pixels; ++i) {
                    ProducerPipeToModule<0>::write(image_data[i]);
                }
            });
        });
        // Launch a module kernel for every module
        // thats... one at the moment
        event e_module = Q.submit([&](handler& h) {
            h.single_task<class Module<0>>([=](){
                size_t sum_pixels = 0;
                for (size_t i = 0; i < num_pixels; ++i) {
                    sum_pixels += ProducerPipeToModule<0>::read();
                }
                result[0] = sum_pixels;
            });
        });
        Q.wait();
        auto cons = event_ms(e_producer);
        auto cons_gbps = (num_pixels * sizeof(H5Read::image_type) / 1e9) / (cons / 1e3);
        fmt::print(" ... consumed in {} ms ({:.3f} Gbps)\n", cons, cons_gbps);

        fmt::print(" ... piped    in {} ms\n", event_ms(e_module));

        auto color = fg(host_sum == result[0] ? fmt::color::green : fmt::color::red);
        fmt::print(color, "     Sum = {} / {}\n", result[0], host_sum);
    }

    free(result, Q);
    free(image_data, Q);
    free(mask_data, Q);
}
