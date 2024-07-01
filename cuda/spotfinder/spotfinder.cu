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

#define VALID_PIXEL 1
#define MASKED_PIXEL 0

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
__device__ float get_distance_from_centre(float x,
                                          float y,
                                          float centre_x,
                                          float centre_y,
                                          float pixel_size_x,
                                          float pixel_size_y) {
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
__device__ float get_resolution(float wavelength,
                                float distance_to_detector,
                                float distance_from_centre) {
    /*
     * Since the angle calculated is, in fact, 2ϴ, we halve to get the
     * proper value of ϴ
    */
    float theta = 0.5 * atanf(distance_from_centre / distance_to_detector);
    return wavelength / (2 * sinf(theta));
}

/**
 * @brief CUDA kernel to apply a resolution mask for an image.
 *
 * This kernel calculates the resolution for each pixel in an image based on the
 * distance from the beam center and the detector properties. It then masks out
 * pixels whose resolution falls outside the specified range [dmin, dmax],
 * provided that the pixel is not already masked, by setting the mask value of
 * the pixel to 0 in the mask data.
 *
 * @param mask Pointer to the mask data indicating valid pixels.
 * @param mask_pitch The pitch (width in bytes) of the mask data.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param wavelength The wavelength of the X-ray beam in Ångströms.
 * @param distance_to_detector The distance from the sample to the detector in mm.
 * @param beam_center_x The x-coordinate of the beam center in the image.
 * @param beam_center_y The y-coordinate of the beam center in the image.
 * @param pixel_size_x The pixel size of the detector in the x-direction in mm.
 * @param pixel_size_y The pixel size of the detector in the y-direction in mm.
 * @param dmin The minimum resolution (d-spacing) threshold.
 * @param dmax The maximum resolution (d-spacing) threshold.
 */
__global__ void apply_resolution_mask(uint8_t *mask,
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
                                      float dmax) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > width || y > height) return;  // Out of bounds

    if (mask[y * mask_pitch + x] == MASKED_PIXEL) {  // Check if the pixel is masked
        /*
        * If the pixel is already masked, we don't need to calculate the
        * resolution for it, so we can just leave it masked
        */
        return;
    }

    float distance_from_centre = get_distance_from_centre(
      x, y, beam_center_x, beam_center_y, pixel_size_x, pixel_size_y);
    float resolution =
      get_resolution(wavelength, distance_to_detector, distance_from_centre);

    // Check if dmin is set and if the resolution is below it
    if (dmin > 0 && resolution < dmin) {
        mask[y * mask_pitch + x] = MASKED_PIXEL;
        return;
    }

    // Check if dmax is set and if the resolution is above it
    if (dmax > 0 && resolution > dmax) {
        mask[y * mask_pitch + x] = MASKED_PIXEL;
        return;
    }

    // If the pixel is not masked and the resolution is within the limits, set the resolution mask to 1
    mask[y * mask_pitch + x] = VALID_PIXEL;
    // ⛔🧊
}

/**
 * @brief Host function to launch the apply_resolution_mask kernel.
 *
 * This function sets up the kernel execution parameters and launches the
 * apply_resolution_mask kernel to generate and apply a resolution mask
 * onto the base mask for the detector.
 *
 * @param blocks The dimensions of the grid of blocks.
 * @param threads The dimensions of the grid of threads within each block.
 * @param shared_memory The size of shared memory required per block (in bytes).
 * @param stream The CUDA stream to execute the kernel.
 * @param mask Device pointer to the mask data indicating valid pixels.
 * @param params The parameters required to calculate the resolution mask.  
 */
void call_apply_resolution_mask(dim3 blocks,
                                dim3 threads,
                                size_t shared_memory,
                                cudaStream_t stream,
                                uint8_t *mask,
                                ResolutionMaskParams params) {
    // Launch the kernel
    apply_resolution_mask<<<blocks, threads, shared_memory, stream>>>(
      mask,
      params.mask_pitch,
      params.width,
      params.height,
      params.wavelength,
      params.detector.distance,
      params.detector.beam_center_x,
      params.detector.beam_center_y,
      params.detector.pixel_size_x,
      params.detector.pixel_size_y,
      params.dmin,
      params.dmax);
}

__global__ void do_spotfinding_dispersion(pixel_t *image,
                                          size_t image_pitch,
                                          uint8_t *mask,
                                          size_t mask_pitch,
                                          int width,
                                          int height,
                                          int kernel_width,
                                          int kernel_height,
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
        for (int row = max(0, y - kernel_height);
             row < min(y + kernel_height + 1, height);
             ++row) {
            int row_offset = image_pitch * row;
            int mask_offset = mask_pitch * row;
            for (int col = max(0, x - kernel_width);
                 col < min(x + kernel_width + 1, width);
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

/**
 * @brief CUDA kernel to erode the mask in place.
 * 
 * This kernel uses shared memory to store a local copy of the mask for each block. It ensures that
 * pixels within a certain radius of a masked pixel are also masked, effectively "eroding" the mask.
 * 
 * @param mask Pointer to the mask data indicating valid pixels.
 * @param mask_pitch The pitch (width in bytes) of the mask data.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param radius The radius around each masked pixel to also be masked.
 */
__global__ void erosion_kernel(uint8_t *mask,
                                        size_t mask_pitch,
                                        int width,
                                        int height,
                                        int radius) {
    // Declare shared memory to store a local copy of the mask for the block
    extern __shared__ uint8_t shared_mask[];

    // Calculate global coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate local coordinates in shared memory
    int local_x = threadIdx.x + radius;
    int local_y = threadIdx.y + radius;

    // Dimensions of shared memory array
    int shared_width = blockDim.x + 2 * radius;
    int shared_height = blockDim.y + 2 * radius;

    // Load central pixels into shared memory
    if (x < width && y < height) {
        shared_mask[local_y * shared_width + local_x] = mask[y * mask_pitch + x];
    } else {
        // If out of bounds, set to MASKED_PIXEL
        shared_mask[local_y * shared_width + local_x] = MASKED_PIXEL;
    }

    // Load border pixels into shared memory
    for (int i = threadIdx.x; i < shared_width; i += blockDim.x) {
        for (int j = threadIdx.y; j < shared_height; j += blockDim.y) {
            int global_x = x + (i - local_x);
            int global_y = y + (j - local_y);
            if (i < radius || i >= shared_width - radius || j < radius
                || j >= shared_height - radius) {
                if (global_x >= 0 && global_x < width && global_y >= 0
                    && global_y < height) {
                    shared_mask[j * shared_width + i] =
                      mask[global_y * mask_pitch + global_x];
                } else {
                    shared_mask[j * shared_width + i] = MASKED_PIXEL;
                }
            }
        }
    }

    // Synchronize threads to ensure all shared memory is loaded
    __syncthreads();

    // If the thread is out of image bounds, exit
    if (x >= width || y >= height) return;

    // Determine if the current pixel should be erased
    bool should_erase = false;
    for (int i = -radius; i <= radius; ++i) {
        for (int j = -radius; j <= radius; ++j) {
            if (shared_mask[(local_y + j) * shared_width + (local_x + i)]
                == MASKED_PIXEL) {
                should_erase = true;
                break;
            }
        }
        if (should_erase) break;
    }

    // Update the mask based on erosion result
    if (should_erase) {
        mask[y * mask_pitch + x] = MASKED_PIXEL;
    }
}
/**
 * @brief CUDA kernel to combine two masks into a third mask.
 * 
 * This kernel combines the original mask and the eroded mask into a third mask using a logical OR operation.
 * 
 * @param original_mask Pointer to the original mask data.
 * @param eroded_mask Pointer to the eroded mask data.
 * @param combined_mask Pointer to the combined mask data.
 * @param mask_pitch The pitch (width in bytes) of the mask data.
 * @param width The width of the image.
 * @param height The height of the image.
 */
__global__ void combine_masks(const uint8_t* original_mask, const uint8_t* eroded_mask, uint8_t* combined_mask, size_t mask_pitch, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    combined_mask[y * mask_pitch + x] = original_mask[y * mask_pitch + x] | eroded_mask[y * mask_pitch + x];
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
    /// One-direction width of kernel. Total kernel span is (K_W * 2 + 1)
    constexpr int BASIC_KERNEL_WIDTH = 3;
    /// One-direction height of kernel. Total kernel span is (K_H * 2 + 1)
    constexpr int BASIC_KERNEL_HEIGHT = 3;

    do_spotfinding_dispersion<<<blocks, threads, shared_memory, stream>>>(
      image,
      image_pitch,
      mask,
      mask_pitch,
      width,
      height,
      BASIC_KERNEL_WIDTH,
      BASIC_KERNEL_HEIGHT,
      result_strong);
}

/**
 * @brief Wrapper function to call the extended spotfinding algorithm.
 * @param blocks The dimensions of the grid of blocks.
 * @param threads The dimensions of the grid of threads within each block.
 * @param shared_memory The size of shared memory required per block (in bytes).
 * @param stream The CUDA stream to execute the kernel.
 * @param image Device pointer to the image data.
 * @param image_pitch The pitch (width in bytes) of the image data.
 * @param mask Device pointer to the mask data indicating valid pixels.
 * @param mask_pitch The pitch (width in bytes) of the mask data.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param result_strong (Output) Device pointer for the strong pixel mask data to be written to.
 */
void call_do_spotfinding_extended(dim3 blocks,
                                  dim3 threads,
                                  size_t shared_memory,
                                  cudaStream_t stream,
                                  pixel_t *image,
                                  size_t image_pitch,
                                  uint8_t *mask,
                                  size_t mask_pitch,
                                  int width,
                                  int height,
                                  uint8_t *result_strong) {
    // Allocate memory for the intermediate result buffer
    uint8_t *d_result_strong_buffer;
    cudaMallocPitch(&d_result_strong_buffer, &mask_pitch, width, height);

    int kernel_radius = 3;

    // Do the first step of spotfinding
    do_spotfinding_dispersion<<<blocks, threads, shared_memory, stream>>>(
      image,
      image_pitch,
      mask,
      mask_pitch,
      width,
      height,
      kernel_radius,  // One-direction width of kernel. Total kernel span is (width * 2 + 1)
      kernel_radius,  // One-direction height of kernel. Total kernel span is (height * 2 + 1)
      d_result_strong_buffer);
    cudaStreamSynchronize(stream);

    {   // Erode the results
        dim3 threads_per_erosion_block(32, 32);
        dim3 erosion_blocks(
        (width + threads_per_erosion_block.x - 1) / threads_per_erosion_block.x,
        (height + threads_per_erosion_block.y - 1) / threads_per_erosion_block.y);

        // Calculate the shared memory size for the erosion kernel
        size_t erosion_shared_memory = (threads_per_erosion_block.x + 2 * kernel_radius)
                                    * (threads_per_erosion_block.y + 2 * kernel_radius)
                                    * sizeof(uint8_t);

        // Perform erosion
        erosion_kernel<<<erosion_blocks,
                                threads_per_erosion_block,
                                erosion_shared_memory,
                                stream>>>(
        d_result_strong_buffer, image_pitch, width, height, kernel_radius);
        cudaStreamSynchronize(stream);
    }

    // Allocate memory for the combined mask
    uint8_t *d_combined_mask;
    cudaMallocPitch(&d_combined_mask, &mask_pitch, width, height);

    {   // Combine mask with positive eroded signals
        dim3 threads_per_combine_block(32, 32);
        dim3 combine_blocks(
        (width + threads_per_combine_block.x - 1) / threads_per_combine_block.x,
        (height + threads_per_combine_block.y - 1) / threads_per_combine_block.y);

        combine_masks<<<combine_blocks,
                        threads_per_combine_block,
                        0,
                        stream>>>(mask, d_result_strong_buffer, d_combined_mask, mask_pitch, width, height);
        cudaStreamSynchronize(stream);
    }
    cudaFree(d_result_strong_buffer);

    kernel_radius = 5;

    // Perform the second step of spotfinding
    do_spotfinding_dispersion<<<blocks, threads, shared_memory, stream>>>(
      image,
      image_pitch,
      d_combined_mask,
      mask_pitch,
      width,
      height,
      kernel_radius,
      kernel_radius,
      result_strong);
    cudaStreamSynchronize(stream);

    cudaFree(d_combined_mask);
}

/**
 * @brief Wrapper function to allow for the selection of the spotfinding algorithm.
 * @param blocks The dimensions of the grid of blocks.
 * @param threads The dimensions of the grid of threads within each block.
 * @param shared_memory The size of shared memory required per block (in bytes).
 * @param stream The CUDA stream to execute the kernel.
 * @param image Device pointer to the image data.
 * @param image_pitch The pitch (width in bytes) of the image data.
 * @param mask Device pointer to the mask data indicating valid pixels.
 * @param mask_pitch The pitch (width in bytes) of the mask data.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param dispersion_algorithm The algorithm to use for spotfinding.
 * @param result_strong (Output) Device pointer for the strong pixel mask data to be written to.
 */
void do_spotfinding(dim3 blocks,
                    dim3 threads,
                    size_t shared_memory,
                    cudaStream_t stream,
                    pixel_t *image,
                    size_t image_pitch,
                    uint8_t *mask,
                    size_t mask_pitch,
                    int width,
                    int height,
                    DispersionAlgorithm dispersion_algorithm,
                    uint8_t *result_strong) {
    void (*)() dispersion_algorithm_call_function = nullptr;  // Function pointer

    switch (dispersion_algorithm) {
    case DispersionAlgorithm::DISPERSION:
        dispersion_algorithm_call_function = call_do_spotfinding_naive;
        break;
    case DispersionAlgorithm::DISPERSION_EXTENDED:
        dispersion_algorithm_call_function = call_do_spotfinding_extended;
        break;
    default:
        throw std::runtime_error("Invalid dispersion algorithm");
    }

    dispersion_algorithm_call_function(blocks,
                                       threads,
                                       shared_memory,
                                       stream,
                                       image,
                                       image_pitch,
                                       mask,
                                       mask_pitch,
                                       width,
                                       height,
                                       result_strong);
}