

#include <fmt/core.h>

#include "common.hpp"

using namespace sycl;

int main(int argc, char **argv) {
    auto parser = FPGAArgumentParser();
    auto &group = parser.add_mutually_exclusive_group(false);
    group.add_argument("--sample")
      .help(
        "Don't load a data file, instead use generated test data. If "
        "H5READ_IMPLICIT_SAMPLE is set, then this is assumed, if a file is not "
        "provided.")
      .implicit_value(true);
    group.add_argument("file")
      .metavar("FILE.nxs")
      .help("Path to the Nexus file to parse")
      .action([](const std::string &value) { fmt::print("Action FILE: {}", value); });

    auto args = parser.parse_args(argc, argv);

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
