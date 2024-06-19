/**
 * Basic Naive Kernel
 * 
 * Does spotfinding in-kernel, without in-depth performance tweaking.
 * 
 */

// #include <bitshuffle.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "spotfinder.h"

namespace cg = cooperative_groups;

/**
 * @brief Function to calculate the distance of a pixel from the beam center.
 * @param x The x-coordinate of the pixel in the image
 * @param y The y-coordinate of the pixel in the image
 * @param center_x The x-coordinate of the pixel beam center in the image
 * @param center_y The y-coordinate of the pixel beam center in the image
 * @param pixel_size_x The pixel size of the detector in the x-direction in mm
 * @param pixel_size_y The pixel size of the detector in the y-direction in mm
 * @return The calculated distance from the beam center in mm
*/
__device__ float get_distance_from_centre(float x, float y, float centre_x, float centre_y, float pixel_size_x, float pixel_size_y) {
    /*
     * Since this calculation is for a broad, general exclusion, we can
     * use basic Pythagoras to calculate the distance from the center.
    */
    float dx = (x - centre_x) * pixel_size_x;
    float dy = (y - centre_y) * pixel_size_y;
    return sqrtf(dx * dx + dy * dy);
}

/**
 * @brief Function to calculate the interplanar distance of a reflection.
 * The interplanar distance is calculated using the formula:
 *         d = λ / (2 * sin(ϴ))
 * @param wavelength The wavelength of the X-ray beam in Å
 * @param distance_to_detector The distance from the sample to the detector in mm
 * @param distance_from_center The distance of the reflection from the beam center in mm
 * @return The calculated d value
*/
__device__ float get_resolution(float wavelength, float distance_to_detector, float distance_from_centre) {
    /*
     * Since the angle calculated is, in fact, 2ϴ, we halve to get the
     * proper value of ϴ
    */
    float theta = 0.5 * atanf(distance_from_centre / distance_to_detector);
    return wavelength / (2 * sinf(theta));
}

__global__ void do_spotfinding_naive(pixel_t *image,
                                     size_t image_pitch,
                                     uint8_t *mask,
                                     size_t mask_pitch,
                                     int width,
                                     int height,
                                     //  int *result_sum,
                                     //  size_t *result_sumsq,
                                     //  uint8_t *result_n,
                                     uint8_t *result_strong) {
    image = image + (image_pitch * height * blockIdx.z);
    // result_sum = result_sum + (image_pitch * height * blockIdx.z);
    // result_sumsq = result_sumsq + (image_pitch * height * blockIdx.z);
    // result_n = result_n + (mask_pitch * height * blockIdx.z);
    result_strong = result_strong + (mask_pitch * height * blockIdx.z);

    auto block = cg::this_thread_block();
    // auto warp = cg::tiled_partition<32>(block);
    // int warpId = warp.meta_group_rank();
    // int lane = warp.thread_rank();

    uint sum = 0;
    size_t sumsq = 0;
    uint8_t n = 0;

    int x = block.group_index().x * block.group_dim().x + block.thread_index().x;
    int y = block.group_index().y * block.group_dim().y + block.thread_index().y;

    // Don't calculate for masked pixels
    bool px_is_valid = mask[y * mask_pitch + x] != 0;
    pixel_t this_pixel = image[y * image_pitch + x];

    if (px_is_valid) {
        for (int row = max(0, y - KERNEL_HEIGHT);
             row < min(y + KERNEL_HEIGHT + 1, height);
             ++row) {
            int row_offset = image_pitch * row;
            int mask_offset = mask_pitch * row;
            for (int col = max(0, x - KERNEL_WIDTH);
                 col < min(x + KERNEL_WIDTH + 1, width);
                 ++col) {
                pixel_t pixel = image[row_offset + col];
                uint8_t mask_pixel = mask[mask_offset + col];
                if (mask_pixel) {
                    sum += pixel;
                    sumsq += pixel * pixel;
                    n += 1;
                }
            }
        }
    }

    if (x < width && y < height) {
        // result_sum[x + image_pitch * y] = sum;
        // result_sumsq[x + image_pitch * y] = sumsq;
        // result_n[x + mask_pitch * y] = n;

        // Calculate the thresholding
        if (px_is_valid) {
            constexpr float n_sig_s = 3.0f;
            constexpr float n_sig_b = 6.0f;

            float sum_f = static_cast<float>(sum);
            float sumsq_f = static_cast<float>(sumsq);

            float mean = sum_f / n;
            float variance = (n * sumsq_f - (sum_f * sum_f)) / (n * (n - 1));
            float dispersion = variance / mean;
            float background_threshold = 1 + n_sig_b * sqrt(2.0f / (n - 1));
            bool not_background = dispersion > background_threshold;
            float signal_threshold = mean + n_sig_s * sqrt(mean);
            bool is_signal = this_pixel > signal_threshold;
            bool is_strong_pixel = not_background && is_signal;
            result_strong[x + mask_pitch * y] = is_strong_pixel;
        } else {
            result_strong[x + mask_pitch * y] = 0;
        }
    }
}
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
                               uint8_t *result_strong) {
    do_spotfinding_naive<<<blocks, threads, shared_memory, stream>>>(
      image, image_pitch, mask, mask_pitch, width, height, result_strong);
}
