

#include <fmt/core.h>

#include "common.hpp"

using namespace sycl;

int main(int argc, char** argv) {
    auto args = FPGAArgumentParser().parse_args(argc, argv);

    return 0;
}