#include <CL/sycl.hpp>
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

    queue Q;
    const size_t num_pixels = modules->slow * modules->fast;
    uint16_t* module_data = malloc_shared<uint16_t>(num_pixels, Q);

    // Print information about the device we are using
    std::cout << "Using Device: " << BOLD
              << Q.get_device().get_info<info::device::name>() << NC << std::endl;

    // Count the zeros in our modules data on-host
    size_t host_zeros = 0;
    for (int i = 0; i < modules->slow; ++i) {
        for (int j = 0; j < modules->fast; ++j) {
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

    int sum = 0;
    int slow = modules->slow;
    int fast = modules->fast;
    auto module_size = range<2>{modules->slow, modules->fast};
    auto module_range = range<2>{64, 64};
    Q.submit([&](handler& h) {
        h.parallel_for(nd_range(module_size, module_range),
                       reduction(&sum, std::plus<>()),
                       [=](nd_item<2> idx, auto& sum) {
                           int y = idx.get_global_id()[0];
                           int x = idx.get_global_id()[1];
                           if (module_data[y * fast + x] == 0) {
                               sum += 1;
                           }
                       });
    });
    Q.wait();
    auto col = sum == host_zeros ? G : R;
    std::cout << "Pixel count from kernel:           " << col << sum << NC << std::endl;

    free(module_data, Q);
    h5read_free_image_modules(modules);
    h5read_free(imagefile);
}
