#pragma once

#include <CL/sycl.hpp>
#include <array>

#include "eiger2xe.h"
#include "h5read.h"

// From https://stackoverflow.com/a/10585543/1118662
constexpr bool is_power_of_two(unsigned int x) {
    return x != 0 && ((x & (x - 1)) == 0);
}

/// One-direction width of kernel. Total kernel span is (K_W * 2 + 1)
constexpr int KERNEL_WIDTH = 3;
/// One-direction height of kernel. Total kernel span is (K_H * 2 + 1)
constexpr int KERNEL_HEIGHT = 3;

// Spotfinding kernel parameters
constexpr float sigma_background = 6;
constexpr float sigma_strong = 4;

constexpr int FULL_KERNEL_HEIGHT = KERNEL_HEIGHT * 2 + 1;

// How many pixels we use at once
constexpr size_t BLOCK_SIZE = 16;

// Module/detector compile-time calculations
/// The number of pixels left over when we divide the image into blocks of BLOCK_SIZE
constexpr size_t BLOCK_REMAINDER = E2XE_16M_FAST % BLOCK_SIZE;
// The number of full blocks that we can fit across an image
constexpr size_t FULL_BLOCKS = (E2XE_16M_FAST - BLOCK_REMAINDER) / BLOCK_SIZE;

constexpr size_t SLOW = E2XE_16M_SLOW;
constexpr size_t FAST = E2XE_16M_FAST;

// Validate these values match our assumptions
static_assert(is_power_of_two(BLOCK_SIZE));

class PipedPixelsArray {
  public:
    typedef H5Read::image_type value_type;

    value_type data[BLOCK_SIZE];

    const value_type& operator[](size_t index) const {
        return this->data[index];
    }
    value_type& operator[](size_t index) {
        return this->data[index];
    }
};

#define DEBUG_IMAGES

struct FindSpotsDebugOutput {
    // H5Read::image_type* image_data;
    H5Read::image_type* sum;
    H5Read::image_type* sumsq;
    float* dispersion;
    float* mean;
    float* variance;
    bool* threshold;

    FindSpotsDebugOutput(sycl::queue Q);
};

auto run_producer(sycl::queue& Q, sycl::host_ptr<uint16_t> image_data) -> sycl::event;

auto run_module(sycl::queue& Q,
                sycl::device_ptr<uint8_t> mask_data,
                sycl::host_ptr<bool> strong_pixels,
                FindSpotsDebugOutput& debug_data) -> sycl::event;