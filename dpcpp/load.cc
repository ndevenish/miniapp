#include <CL/sycl.hpp>
#include <iostream>

#include "h5read.h"

using namespace sycl;

int main(int argc, char** argv) {
    // Read a single module from a supplied file
    auto imagefile = h5read_parse_standard_args(argc, argv);
    auto modules = h5read_get_image_modules(imagefile, 0);

    queue Q;
    const size_t num_pixels = modules->slow * modules->fast;
    uint16_t* module_data = malloc_shared<uint16_t>(num_pixels, Q);

    // Count the zeros in our modules data
    size_t host_zeros = 0;
    for (int i = 0; i < modules->slow; ++i) {
        for (int j = 0; j < modules->fast; ++j) {
            if (modules->data[i * modules->fast + j] == 0) {
                host_zeros += 1;
            }
        }
    }
    std::cout << "Host count zeros for first module: " << host_zeros << std::endl;

    // Copy our module to the shared buffer
    std::copy(modules->data, modules->data + num_pixels, module_data);

    // Only need to migrate host/device memory - not shared
    // Q.submit(|&}(handler &h) {
    //     h.memcpy(module_data, )
    // });
    // Q.submit(|&|(handler *h) {

    // })

    free(module_data, Q);
    h5read_free_image_modules(modules);
    h5read_free(imagefile);
}
