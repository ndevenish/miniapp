#include "spotfinder.h"

namespace cg = cooperative_groups;

/**
 * @brief Structure to store thread-specific information for the erosion kernel.
 */
struct KernelThreadParams {
    int x;
    int y;
    int local_x;
    int local_y;
    int shared_width;
    int shared_height;
};

/**
 * @brief Calculate block information for the current thread.
 * @param block The cooperative group for the current block.
 * @param radius The radius each thread should consider around a pixel
 * @return KernelThreadParams structure containing thread-specific information.
 */
__device__ KernelThreadParams calculate_block_info(cg::thread_block block, int radius) {
    KernelThreadParams threadParams;

    // Calculate global coordinates
    threadParams.x =
      block.group_index().x * block.group_dim().x + block.thread_index().x;
    threadParams.y =
      block.group_index().y * block.group_dim().y + block.thread_index().y;

    // Calculate local coordinates in shared memory
    threadParams.local_x = block.thread_index().x + radius;
    threadParams.local_y = block.thread_index().y + radius;

    // Calculate shared memory dimensions
    threadParams.shared_width = block.group_dim().x + 2 * radius;
    threadParams.shared_height = block.group_dim().y + 2 * radius;

    return threadParams;
}

/**
 * @brief Load central pixels into shared memory.
 * @param block The cooperative group for the current block.
 * @param threadParams Thread-specific information for the current thread.
 * @param mask Pointer to the mask data.
 * @param shared_mask Pointer to the shared memory buffer.
 * @param mask_pitch The pitch (width in bytes) of the mask data.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param radius The radius around each masked pixel to be considered.
 */
__device__ void load_central_pixels(cg::thread_block block,
                                    const KernelThreadParams &threadParams,
                                    const uint8_t *mask,
                                    uint8_t *shared_mask,
                                    size_t mask_pitch,
                                    int width,
                                    int height,
                                    int radius) {
    // Load central pixels into shared memory
    if (threadParams.x < width && threadParams.y < height) {
        shared_mask[threadParams.local_y * threadParams.shared_width
                    + threadParams.local_x] =
          mask[threadParams.y * mask_pitch + threadParams.x];
    } else {
        shared_mask[threadParams.local_y * threadParams.shared_width
                    + threadParams.local_x] = MASKED_PIXEL;
    }
}

/**
 * @brief Load border pixels into shared memory.
 * @param block The cooperative group for the current block.
 * @param threadParams Thread-specific information for the current thread.
 * @param mask Pointer to the mask data.
 * @param shared_mask Pointer to the shared memory buffer.
 * @param mask_pitch The pitch (width in bytes) of the mask data.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param radius The radius around each masked pixel to be considered. 
 */
__device__ void load_border_pixels(cg::thread_block block,
                                   const KernelThreadParams &threadParams,
                                   const uint8_t *mask,
                                   uint8_t *shared_mask,
                                   size_t mask_pitch,
                                   int width,
                                   int height,
                                   int radius) {
    // Load border pixels into shared memory
    for (int i = block.thread_index().x; i < threadParams.shared_width;
         i += block.group_dim().x) {
        for (int j = block.thread_index().y; j < threadParams.shared_height;
             j += block.group_dim().y) {
            int global_x = threadParams.x + (i - threadParams.local_x);
            int global_y = threadParams.y + (j - threadParams.local_y);

            bool is_within_central_region =
              (i >= radius && i < threadParams.shared_width - radius && j >= radius
               && j < threadParams.shared_height - radius);
            bool is_global_x_in_bounds = (global_x >= 0 && global_x < width);
            bool is_global_y_in_bounds = (global_y >= 0 && global_y < height);

            if (is_within_central_region) {
                continue;
            }

            if (is_global_x_in_bounds && is_global_y_in_bounds) {
                shared_mask[j * threadParams.shared_width + i] =
                  mask[global_y * mask_pitch + global_x];
            } else {
                shared_mask[j * threadParams.shared_width + i] = MASKED_PIXEL;
            }
        }
    }
}

/**
 * @brief Determine if the current pixel should be erased based on the mask.
 * @param threadParams Thread-specific information for the current thread.
 * @param shared_mask Pointer to the shared memory buffer.
 * @param radius The radius around each masked pixel to be considered.
 * @param distance_threshold The maximum Chebyshev distance for erasing the current pixel.
 * @return True if the current pixel should be erased, false otherwise.
 */
__device__ bool determine_erasure(const KernelThreadParams &threadParams,
                                  const uint8_t *shared_mask,
                                  int radius,
                                  int distance_threshold) {
    bool should_erase = false;
    for (int i = -radius; i <= radius; ++i) {
        for (int j = -radius; j <= radius; ++j) {
            if (shared_mask[(threadParams.local_y + j) * threadParams.shared_width
                            + (threadParams.local_x + i)]
                == MASKED_PIXEL) {
                int chebyshev_distance = max(abs(i), abs(j));
                if (chebyshev_distance <= distance_threshold) {
                    should_erase = true;
                    break;
                }
            }
        }
        if (should_erase) break;
    }
    return should_erase;
}

__global__ void determine_erasure_kernel(const uint8_t *shared_mask,
                                         int shared_width,
                                         int local_x,
                                         int local_y,
                                         int radius,
                                         int distance_threshold,
                                         unsigned int *should_erase) {
    int i = threadIdx.x - radius;
    int j = threadIdx.y - radius;

    if (shared_mask[(local_y + j) * shared_width + (local_x + i)] == MASKED_PIXEL) {
        int chebyshev_distance = max(abs(i), abs(j));
        if (chebyshev_distance <= distance_threshold) {
            atomicExch(should_erase, 1u);
        }
    }
}

/**
 * @brief Device function to determine if the current pixel should be erased using dynamic parallelism.
 * @param shared_mask Pointer to the shared memory buffer.
 * @param threadParams Thread-specific information for the current thread.
 * @param radius The radius around each masked pixel to be considered.
 * @param distance_threshold The maximum Chebyshev distance for erasing the current pixel.
 * @return True if the current pixel should be erased, false otherwise.
 */
__device__ bool launch_determine_erasure_kernel(const uint8_t *shared_mask,
                                                const KernelThreadParams &threadParams,
                                                int radius,
                                                int distance_threshold) {
    // Allocate memory for the erasure flag
    unsigned int *d_should_erase;
    cudaMalloc(&d_should_erase, sizeof(unsigned int));
    cudaMemset(d_should_erase, 0, sizeof(unsigned int));

    // Launch the erasure determination kernel
    dim3 erasure_block_size(2 * radius + 1, 2 * radius + 1);
    determine_erasure_kernel<<<1, erasure_block_size>>>(shared_mask,
                                                        threadParams.shared_width,
                                                        threadParams.local_x,
                                                        threadParams.local_y,
                                                        radius,
                                                        distance_threshold,
                                                        d_should_erase);

    // Copy the result back to the host
    unsigned int h_should_erase_uint;
    cudaMemcpy(&h_should_erase_uint,
               d_should_erase,
               sizeof(unsigned int),
               cudaMemcpyDeviceToHost);
    cudaFree(d_should_erase);

    return h_should_erase_uint == 1u;
}

/**
 * @brief CUDA kernel to apply erosion based on the mask and update the erosion_mask.
 * 
 * This kernel uses shared memory to store a local copy of the mask for each block.
 * 
 * @param mask Pointer to the mask data indicating valid pixels.
 * @param erosion_mask Pointer to the allocated output erosion mask.
 * @param mask_pitch The pitch (width in bytes) of the mask data.
 * @param erosion_mask Pointer to the output erosion mask data. (Expected to be the same size as the mask)
 * @param width The width of the image.
 * @param height The height of the image.
 * @param radius The radius around each masked pixel to also be masked.
 */
__global__ void erosion_kernel(
  const uint8_t __restrict__ *mask,
  uint8_t __restrict__ *erosion_mask,
  // __restrict__ is a hint to the compiler that the two pointers are not
  // aliased, allowing the compiler to perform more agressive optimizations
  size_t mask_pitch,
  int width,
  int height,
  int radius) {
    // Declare shared memory to store a local copy of the mask for the block
    extern __shared__ uint8_t shared_mask[];

    // Create a cooperative group for the current block
    cg::thread_block block = cg::this_thread_block();

    // Calculate block information
    KernelThreadParams threadParams = calculate_block_info(block, radius);

    // Load central pixels
    load_central_pixels(
      block, threadParams, mask, shared_mask, mask_pitch, width, height, radius);

    // Load border pixels
    load_border_pixels(
      block, threadParams, mask, shared_mask, mask_pitch, width, height, radius);

    // Synchronize threads to ensure all shared memory is loaded
    block.sync();

    /*
     * If the current pixel is outside the image bounds, return without doing anything.
     * We do this after loading shared memory as it may be necessary for this thread 
     * to load border pixels.
    */
    if (threadParams.x >= width || threadParams.y >= height) return;

    // Determine if the current pixel should be erased
    bool should_erase =
      determine_erasure(threadParams,
                        shared_mask,
                        radius,
                        2);  // Use 2 as the Chebyshev distance threshold
    // dynamic parrelism based
    bool should_erase =
      launch_determine_erasure_kernel(shared_mask,
                                      threadParams,
                                      radius,
                                      2);  // Use 2 as the Chebyshev distance threshold

    // Update the erosion_mask based on erosion result
    if (should_erase) {
        erosion_mask[threadParams.y * mask_pitch + threadParams.x] = MASKED_PIXEL;
    } else {
        erosion_mask[threadParams.y * mask_pitch + threadParams.x] = VALID_PIXEL;
    }
}