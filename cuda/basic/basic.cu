#include <fmt/core.h>

#include "common.hpp"
#include "h5read.h"

using namespace fmt;

int main(int argc, char** argv) {
    // Parse arguments and get our H5Reader
    auto parser = CUDAArgumentParser();
    parser.add_h5read_arguments();
    auto args = parser.parse_args(argc, argv);

    auto reader = args.file.empty() ? H5Read() : H5Read(args.file);

    // // Get the name of the device we are using
    // cudaDeviceProp deviceProp;
    // cudaGetDeviceProperties(&deviceProp, args.device_index);
    // cuda_throw_error();

    // int deviceCount;
    // cudaGetDeviceCount(&deviceCount);

    // print("Enumerated {} devices\n", bold(deviceCount));
    // for (int device = 0; device < deviceCount; ++device) {
    //     cudaDeviceProp deviceProp;
    //     cudaGetDeviceProperties(&deviceProp, device);
    //     print("{}: {}\n", device, blue(bold(deviceProp.name)));
    //     print("        Compute Capability: {}\n",
    //           bold("{}.{}", deviceProp.major, deviceProp.minor));
    //     print("             Max Grid Size: {}\n",
    //           bold("{} x {} x {}",
    //                deviceProp.maxGridSize[0],
    //                deviceProp.maxGridSize[1],
    //                deviceProp.maxGridSize[2]));
    //     print("    Max threads in a block: {}\n", bold(deviceProp.maxThreadsPerBlock));
    // }
}
