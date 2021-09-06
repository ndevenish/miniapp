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
    // Read a single module from a supplied file
    auto imagefile = h5read_parse_standard_args(argc, argv);
    auto modules = h5read_get_image_modules(imagefile, 0);

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

    const size_t num_pixels = modules->slow * modules->fast;
    uint16_t* module_data = malloc_shared<uint16_t>(num_pixels, Q);

    // Print information about the device we are using
    std::string device_kind = Q.get_device().is_cpu()           ? "CPU"
                              : Q.get_device().is_gpu()         ? "GPU"
                              : Q.get_device().is_accelerator() ? "FPGA"
                                                                : "Unknown";
    std::cout << "Using " << BOLD << device_kind << NC << " Device: " << BOLD
              << Q.get_device().get_info<info::device::name>() << NC << std::endl;

    std::cout << "Module size s,f: " << modules->slow << ", " << modules->fast
              << std::endl;

    // Count the zeros in our modules data on-host
    size_t host_zeros = 0;
    for (int i = 0; i < std::min(modules->slow, (size_t)512); ++i) {
        for (int j = 0; j < std::min(modules->fast, (size_t)1024); ++j) {
            if (modules->data[i * modules->fast + j] == 0) {
                host_zeros += 1;
            }
        }
    }
    std::cout << "Number of pixels in module:        " << modules->slow * modules->fast
              << std::endl;
    std::cout << "Host count zeros for first module: " << host_zeros << std::endl;

    // Copy our module to the shared buffer
    std::copy(modules->data, modules->data + num_pixels, module_data);

    // int sum = 0;
    // buffer buf_sum(&sum);
    // int slow = modules->slow;
    // int fast = modules->fast;
    //     auto module_size = range<2>{512, 1024};  // modules->slow, modules->fast};
    // #ifdef FPGA
    //     auto module_range = module_size;
    // #else
    //     auto module_range = range<2>{64, 64};
    // #endif
    //     Q.submit([&](handler& h) {
    //         h.parallel_for(nd_range(module_size, module_range), [=](nd_item<2> idx) {
    //             accessor sum(buf_sum, h, write_only);
    //             int y = idx.get_global_id()[0];
    //             int x = idx.get_global_id()[1];
    //             if (module_data[y * fast + x] == 0) {
    //                 sum[0] += 1;
    //             }
    //         });
    //     });
    //     Q.wait();
    //     auto col = sum == host_zeros ? G : R;
    //     std::cout << "Pixel count from kernel:           " << col << sum << NC <<
    //     std::endl;

    free(module_data, Q);
    h5read_free_image_modules(modules);
    h5read_free(imagefile);
}
