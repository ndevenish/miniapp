#ifndef _DPCPP_COMMON_H
#define _DPCPP_COMMON_H

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

constexpr auto R = "\033[31m";
constexpr auto G = "\033[32m";
constexpr auto Y = "\033[33m";
constexpr auto B = "\033[34m";
constexpr auto GRAY = "\033[37m";
constexpr auto BOLD = "\033[1m";
constexpr auto NC = "\033[0m";

sycl::queue initialize_queue() {
#ifdef FPGA
// Select either:
//  - the FPGA emulator device (CPU emulation of the FPGA)
//  - the FPGA device (a real FPGA)
#if defined(FPGA_EMULATOR)
    sycl::INTEL::fpga_emulator_selector device_selector;
#else
    sycl::INTEL::fpga_selector device_selector;
#endif
    sycl::queue Q(device_selector, sycl::property::queue::enable_profiling{});
#else
    sycl::queue Q{sycl::property::queue::enable_profiling{}};
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
               Q.get_device().get_info<sycl::info::device::name>());
    return Q;
}

#endif