#include <fmt/core.h>

#include "common.hpp"
#include "h5read.h"

using namespace fmt;

using pixel_t = H5Read::image_type;

__global__ void do_sum_image(size_t *dest,
                             pixel_t *data,
                             size_t pitch,
                             size_t width,
                             size_t height) {}

int main(int argc, char **argv) {
    // Parse arguments and get our H5Reader
    auto parser = CUDAArgumentParser();
    parser.add_h5read_arguments();
    auto args = parser.parse_args(argc, argv);

    auto reader = args.file.empty() ? H5Read() : H5Read(args.file);

    auto image_shape = reader.image_shape();
    H5Read::image_type *host_image = nullptr;
    // Create a host memory area to store the current image
    cudaMallocHost(&host_image, image_shape[0] * image_shape[1] * sizeof(pixel_t));
    // Create a device-side pitched area
    image_t *dev_image = nullptr;
    size_t *dev_result = nullptr;

    size_t device_pitch = 0;
    cudaMallocPitch(&dev_image,
                    &device_pitch,
                    image_shape[1] * sizeof(pixel_t),
                    image_shape[0] * sizeof(pixel_t));
    cudaMalloc(&dev_result, sizeof(dev_result));

    cuda_throw_error();
    print("Allocated device memory. Pitch = {} vs naive {}\n",
          device_pitch,
          image_shape[1] * sizeof(pixel_t));

    for (size_t image_id = 0; image_id < reader.get_number_of_images(); ++image_id) {
        print("Image {}:\n", image_id);
        reader.get_image_into(image_id, host_image);

        // Calculate the sum of all pixels host-side
        uint32_t sum = 0;
        for (int y = 0; y < image_shape[0]; ++y) {
            for (int x = 0; x < image_shape[1]; ++x) {
                sum += host_image[x + y * image_shape[1]];
            }
        }
        print("    Summed pixels: {}\n", bold(sum));

        // Copy to device
        cudaMemcpy2D(dev_image,
                     device_pitch,
                     dev_image,
                     image_shape[1] * sizeof(pixel_t),
                     image_shape[1] * sizeof(pixel_t),
                     image_shape[2] * sizeof(pixel_t),
                     cudaMemcpyHostToDevice);
    }
    cudaFree(dev_result);
    cudaFree(dev_image);
    cudaFreeHost(host_image);
}
