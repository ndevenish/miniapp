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
    queue Q(device_selector);  //, dpc_common::exception_handler);
#else
    queue Q;  //, dpc_common::exception_handler);
#endif

    // Print information about the device we are using
    std::string device_kind = Q.get_device().is_cpu()           ? "CPU"
                              : Q.get_device().is_gpu()         ? "GPU"
                              : Q.get_device().is_accelerator() ? "FPGA"
                                                                : "Unknown";
    std::cout << "Using " << BOLD << device_kind << NC << " Device: " << BOLD
              << Q.get_device().get_info<info::device::name>() << NC << std::endl;

    // Read a single module from a supplied file
    auto modules = reader.get_image_modules(0);
    const size_t num_pixels = modules.slow * modules.fast;
    uint16_t* module_data = malloc_shared<uint16_t>(num_pixels, Q);

    std::cout << "Module size s,f: " << modules.slow << ", " << modules.fast
              << std::endl;

    // Count the zeros in our modules data on-host
    size_t host_zeros = 0;
    for (int i = 0; i < std::min(modules.slow, (size_t)512); ++i) {
        for (int j = 0; j < std::min(modules.fast, (size_t)1024); ++j) {
            if (modules.data[i * modules.fast + j] == 0) {
                host_zeros += 1;
            }
        }
    }
    std::cout << "Number of pixels in module:        " << modules.slow * modules.fast
              << std::endl;
    std::cout << "Host count zeros for first module: " << host_zeros << std::endl;

    // Copy our module to the shared buffer
    std::copy(modules.data, modules.data + num_pixels, module_data);

    auto fast = modules.fast;
    auto slow = modules.slow;
    auto module_size = range<2>{512, 1024};
    auto module_range = range<2>{128, 128};
    const int num_blocks =
      module_size[0] / module_range[0] * module_size[1] / module_range[1];
    std::cout << "Number of separate ranges: " << num_blocks << std::endl;

    // uint32_t* interim_sum = malloc_device<uint32_t>(num_blocks, Q);
    auto result = malloc_shared<uint32_t>(1, Q);
    *result = 0;
    Q.submit([&](handler& h) {
        h.single_task<class QRD>([=]() {
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

    uint32_t kernel_zeros = result[0];
    auto color = fg(kernel_zeros == host_zeros ? fmt::color::green : fmt::color::red);
    fmt::print(color, "{:35}", "Result from kernel:");

    free(result, Q.get_context());
    free(module_data, Q.get_context());
}
