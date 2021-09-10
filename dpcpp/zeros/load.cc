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

class ZeroCounter;

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
    fmt::print("Using {0}{2}{1} Device: {0}{3}{1}\n",
               BOLD,
               NC,
               device_kind,
               Q.get_device().get_info<info::device::name>());

    // Read a single module from a supplied file
    auto modules = reader.get_image_modules(0);
    const size_t num_pixels = modules.slow * modules.fast;
    uint16_t* module_data = malloc_shared<uint16_t>(num_pixels, Q);
    bool* module_mask = malloc_shared<bool>(num_pixels, Q);

    fmt::print("Module size s,f: {}, {}\n", modules.slow, modules.fast);

    // Count the zeros in our modules data on-host
    size_t host_zeros = 0;
    for (auto pixel : modules.modules[0]) {
        if (pixel == 0) host_zeros++;
    }

    // for (int i = 0; i < modules.slow; ++i) {
    //     for (int j = 0; j < modules.fast; ++j) {
    //         if (modules.data[i * modules.fast + j] == 0) {
    //             host_zeros += 1;
    //         }
    //     }
    // }
    fmt::print("Number of pixels in module:        {}\n", modules.slow * modules.fast);
    fmt::print("Host count zeros for first module: {}\n", host_zeros);

    // Copy our module to the shared buffer
    std::copy(modules.modules[0].begin(), modules.modules[0].end(), module_data);
    std::copy(modules.modules[0].begin(), modules.modules[0].end(), module_mask);

    auto fast = modules.fast;
    auto slow = modules.slow;

    auto result = malloc_shared<uint32_t>(1, Q);
    *result = 0;

    event e = Q.submit([&](handler& h) {
        h.single_task<class ZeroCounter>([=]() {
            int zeros = 0;
            for (int i = 0; i < num_pixels; ++i) {
                if (module_data[i] == 0) {
                    zeros++;
                }
            }
            result[0] = zeros;
        });
    });
    Q.wait();
    double start = e.get_profiling_info<info::event_profiling::command_start>();
    double end = e.get_profiling_info<info::event_profiling::command_end>();
    double kernel_time = (double)(end - start) * 1e-6;

    uint32_t kernel_zeros = result[0];
    auto color = fg(kernel_zeros == host_zeros ? fmt::color::green : fmt::color::red);
    fmt::print(color, "{:34} {}\n", "Result from kernel:", kernel_zeros);

    fmt::print("In: {}ms\n", kernel_time);

    free(result, Q.get_context());
    free(module_data, Q.get_context());
}
