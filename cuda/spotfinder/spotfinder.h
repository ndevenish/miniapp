#ifndef SPOTFINDER_H
#define SPOTFINDER_H

#include <builtin_types.h>

#include "h5read.h"

using pixel_t = H5Read::image_type;

/// One-direction width of kernel. Total kernel span is (K_W * 2 + 1)
constexpr int KERNEL_WIDTH = 3;
/// One-direction height of kernel. Total kernel span is (K_H * 2 + 1)
constexpr int KERNEL_HEIGHT = 3;

void call_apply_resolution_mask(dim3 blocks,
                                dim3 threads,
                                size_t shared_memory,
                                cudaStream_t stream,
                                uint8_t *mask,
                                size_t mask_pitch,
                                int width,
                                int height,
                                float wavelength,
                                float distance_to_detector,
                                float beam_center_x,
                                float beam_center_y,
                                float pixel_size_x,
                                float pixel_size_y,
                                float dmin,
                                float dmax);

void call_do_spotfinding_naive(dim3 blocks,
                               dim3 threads,
                               size_t shared_memory,
                               cudaStream_t stream,
                               pixel_t *image,
                               size_t image_pitch,
                               uint8_t *mask,
                               size_t mask_pitch,
                               int width,
                               int height,
                               //  int *result_sum,
                               //  size_t *result_sumsq,
                               //  uint8_t *result_n,
                               uint8_t *result_strong);

#endif