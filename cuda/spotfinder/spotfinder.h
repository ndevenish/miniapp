#ifndef SPOTFINDER_H
#define SPOTFINDER_H

#include <builtin_types.h>
#include <nlohmann/json.hpp>

#include "h5read.h"

using pixel_t = H5Read::image_type;

/// One-direction width of kernel. Total kernel span is (K_W * 2 + 1)
constexpr int KERNEL_WIDTH = 3;
/// One-direction height of kernel. Total kernel span is (K_H * 2 + 1)
constexpr int KERNEL_HEIGHT = 3;

/**
 * @brief Struct to store the geometry of the detector.
*/
struct detector_geometry {
    float pixel_size_x;
    float pixel_size_y;
    float beam_center_x;
    float beam_center_y;
    float distance;

    /**
     * @brief Constructor to initialize the detector geometry from a JSON object.
     * @param geometry_data A JSON object containing the detector geometry data.
     * The JSON object must have the following keys:
     * - pixel_size_x: The pixel size of the detector in the x-direction in mm
     * - pixel_size_y: The pixel size of the detector in the y-direction in mm
     * - beam_center_x: The x-coordinate of the pixel beam center in the image
     * - beam_center_y: The y-coordinate of the pixel beam center in the image
     * - distance: The distance from the sample to the detector in mm
    */
    detector_geometry(nlohmann::json geometry_data) {
        std::vector<std::string> required_keys = {
          "pixel_size_x", "pixel_size_y", "beam_center_x", "beam_center_y", "distance"};

        for (const auto &key : required_keys) {
            if (geometry_data.find(key) == geometry_data.end()) {
                throw std::invalid_argument("Key " + key
                                            + " is missing from the input JSON");
            }
        }

        pixel_size_x = geometry_data["pixel_size_x"];
        pixel_size_y = geometry_data["pixel_size_y"];
        beam_center_x = geometry_data["beam_center_x"];
        beam_center_y = geometry_data["beam_center_y"];
        distance = geometry_data["distance"];
    }
};

/**
 * @brief Struct to store parameters for calculating the resolution filtered mask
*/
struct ResolutionMaskParams {
    size_t mask_pitch;
    int width;
    int height;
    float wavelength;
    detector_geometry detector;
    float dmin;
    float dmax;
};

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