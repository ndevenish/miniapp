

#include <fmt/core.h>

#include "common.hpp"
#include "h5read.h"

using namespace sycl;

int main(int argc, char **argv) {
    auto parser = FPGAArgumentParser();
    parser.add_h5read_arguments();
    auto args = parser.parse_args(argc, argv);
    auto reader = args.file.empty() ? H5Read() : H5Read(args.file);

    fmt::print("Number of images: {}\n", reader.get_number_of_images());
    return 0;
}

/*
      "Options:\n\
  FILE.nxs      Path to the Nexus file to parse\n\
  -h, --help    Show this message\n\
  -v            Verbose HDF5 message output\n\
  --sample      Don't load a data file, instead use generated test data.\n\
                If H5READ_IMPLICIT_SAMPLE is set, then this is assumed,\n\
                if a file is not provided.";
*/
