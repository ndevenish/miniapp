
#include "baseline.h"
#include "eiger2xe.h"

#include <dials/algorithms/image/filter/distance.h>
#include <dials/algorithms/image/filter/index_of_dispersion_filter.h>
#include <dials/algorithms/image/filter/mean_and_variance.h>
#include <dials/error.h>
#include <scitbx/array_family/ref_reductions.h>
#include <scitbx/array_family/tiny_types.h>
#include <omp.h>
#include <stdio.h>

#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>

#include "spotfind_test_utils.h"

namespace baseline {

// using namespace dials;
using namespace dials::algorithms;

/**
 * A class to compute the threshold using index of dispersion
 */
class DispersionThreshold {
  public:
    /**
     * Enable more efficient memory usage by putting components required for the
     * summed area table closer together in memory
     */
    template <typename T>
    struct Data {
        int m;
        T x;
        T y;
    };

    DispersionThreshold(int2 image_size,
                        int2 kernel_size,
                        double nsig_b,
                        double nsig_s,
                        double threshold,
                        int min_count)
        : image_size_(image_size),
          kernel_size_(kernel_size),
          nsig_b_(nsig_b),
          nsig_s_(nsig_s),
          threshold_(threshold),
          min_count_(min_count) {
        // Check the input
        DIALS_ASSERT(threshold_ >= 0);
        DIALS_ASSERT(nsig_b >= 0 && nsig_s >= 0);
        DIALS_ASSERT(image_size.all_gt(0));
        DIALS_ASSERT(kernel_size.all_gt(0));

        // Ensure the min counts are valid
        std::size_t num_kernel = (2 * kernel_size[0] + 1) * (2 * kernel_size[1] + 1);
        if (min_count_ <= 0) {
            min_count_ = num_kernel;
        } else {
            DIALS_ASSERT(min_count_ <= num_kernel && min_count_ > 1);
        }

        // Allocate the buffer
        std::size_t element_size = sizeof(Data<double>);
        buffer_.resize(element_size * image_size[0] * image_size[1]);
    }

    /**
     * Compute the summed area tables for the mask, src and src^2.
     * @param src The input array
     * @param mask The mask array
     */
    template <typename T>
    double compute_sat(af::ref<Data<T>> table,
                     const af::const_ref<T, af::c_grid<2>> &src,
                     const af::const_ref<bool, af::c_grid<2>> &mask) {
        double t0 = omp_get_wtime();
        // Largest value to consider
        const T BIG = (1 << 24);  // About 16m counts

        // Get the size of the image
        std::size_t ysize = src.accessor()[1]; // 1028/4148 = FAST
        std::size_t xsize = src.accessor()[0]; // 512/4362 = SLOW

        // Create the summed area table
        for (std::size_t j = 0, k = 0; j < ysize; ++j) {
            int m = 0;
            T x = 0;
            T y = 0;
            for (std::size_t i = 0; i < xsize; ++i, ++k) {
                int mm = (mask[k] && src[k] < BIG) ? 1 : 0;
                m += mm;
                x += mm * src[k];
                y += mm * src[k] * src[k];
                if (j == 0) {
                    table[k].m = m;
                    table[k].x = x;
                    table[k].y = y;
                } else {
                    table[k].m = table[k - xsize].m + m;
                    table[k].x = table[k - xsize].x + x;
                    table[k].y = table[k - xsize].y + y;
                }
            }
        }
        return omp_get_wtime() - t0;
    }

    template <typename T>
    double compute_sat_new(af::ref<Data<T>> table,
                     const af::const_ref<T, af::c_grid<2>> &src,
                     const af::const_ref<bool, af::c_grid<2>> &mask) {
        double t0 = omp_get_wtime();
        // Largest value to consider
        const T BIG = (1 << 24);  // About 16m counts

        // Get the size of the image
        std::size_t ysize = src.accessor()[1];
        std::size_t xsize = src.accessor()[0];

        // Create the summed area table
        for (size_t module_row_num=0; module_row_num<8; module_row_num++) {
            size_t row_offset = module_row_num * (E2XE_GAP_SLOW+E2XE_MOD_SLOW) * E2XE_16M_FAST;
            size_t offset2 = 1 * (E2XE_MOD_FAST+E2XE_GAP_FAST);
            size_t offset3 = 2 * (E2XE_MOD_FAST+E2XE_GAP_FAST);
            size_t offset4 = 3 * (E2XE_MOD_FAST+E2XE_GAP_FAST);
            size_t row_end = (module_row_num == 7) ? 512 : 550;
            size_t k =row_offset;
            for (std::size_t j = 0; j < 512; ++j) {
                k = row_offset + j * E2XE_16M_FAST;
                int m = 0;
                T x = 0;
                T y = 0;
                int m2 = 0;
                T x2 = 0;
                T y2 = 0;
                int m3 = 0;
                T x3 = 0;
                T y3 = 0;
                int m4 = 0;
                T x4 = 0;
                T y4 = 0;
                for (std::size_t i = 0; i < 1028; ++i, ++k) {

                    size_t k2 = k+offset2;
                    size_t k3 = k+offset3;
                    size_t k4 = k+offset4;

                    int mm = (mask[k] && src[k] < BIG) ? 1 : 0;
                    m += mm;
                    x += mm * src[k];
                    y += mm * src[k] * src[k];

                    int mm2= (mask[k2] && src[k2] < BIG) ? 1 : 0;
                    m2 += mm2;
                    x2 += mm2 * src[k2];
                    y2 += mm2 * src[k2] * src[k2];

                    int mm3= (mask[k3] && src[k3] < BIG) ? 1 : 0;
                    m3 += mm3;
                    x3 += mm3 * src[k3];
                    y3 += mm3 * src[k3] * src[k3];

                    int mm4= (mask[k4] && src[k4] < BIG) ? 1 : 0;
                    m4 += mm4;
                    x4 += mm4 * src[k4];
                    y4 += mm4 * src[k4] * src[k4];

                    if (j == 0 && module_row_num==0) {
                        table[k].m = m;
                        table[k].x = x;
                        table[k].y = y;

                        table[k2].m = m2;
                        table[k2].x = x2;
                        table[k2].y = y2;

                        table[k3].m = m3;
                        table[k3].x = x3;
                        table[k3].y = y3;

                        table[k4].m = m4;
                        table[k4].x = x4;
                        table[k4].y = y4;
                    } else {
                        table[k].m = table[k - xsize].m + m;
                        table[k].x = table[k - xsize].x + x;
                        table[k].y = table[k - xsize].y + y;

                        table[k2].m = table[k2 - xsize].m + m2;
                        table[k2].x = table[k2 - xsize].x + x2;
                        table[k2].y = table[k2 - xsize].y + y2;

                        table[k3].m = table[k3 - xsize].m + m3;
                        table[k3].x = table[k3 - xsize].x + x3;
                        table[k3].y = table[k3 - xsize].y + y3;

                        table[k4].m = table[k4 - xsize].m + m4;
                        table[k4].x = table[k4 - xsize].x + x4;
                        table[k4].y = table[k4 - xsize].y + y4;
                    }
                }
            }
        }
        return omp_get_wtime() - t0;
    }

    /**
     * Compute the threshold
     * @param src - The input array
     * @param mask - The mask array
     * @param dst The output array
     */

    template <typename T>
    double compute_threshold(af::ref<Data<T>> table,
                           const af::const_ref<T, af::c_grid<2>> &src,
                           const af::const_ref<bool, af::c_grid<2>> &mask,
                           af::ref<bool, af::c_grid<2>> dst) {
        double t0 = omp_get_wtime();
        // Get the size of the image
        // I HAVE SWAPPED THESE TO MATCH THE DATA INDICES
        std::size_t ysize = src.accessor()[1];
        std::size_t xsize = src.accessor()[0];

        // The kernel size
        int kxsize = kernel_size_[1];
        int kysize = kernel_size_[0];

        // Calculate the local mean at every point
//#pragma omp parallel for default(none) shared(mask, dst, min_count_, src, threshold_, nsig_b_, nsig_s_, table, xsize, ysize, kxsize, kysize) //private(k)
        for (std::size_t j = 0; j < ysize; ++j) {
            for (std::size_t i = 0; i < xsize; ++i) {
                size_t k = j*xsize+i;
                int i0 = i - kxsize - 1, i1 = i + kxsize;
                int j0 = j - kysize - 1, j1 = j + kysize;
                i1 = i1 < xsize ? i1 : xsize - 1;
                j1 = j1 < ysize ? j1 : ysize - 1;
                int k0 = j0 * xsize;
                int k1 = j1 * xsize;

                // Compute the number of points valid in the local area,
                // the sum of the pixel values and the sum of the squared pixel
                // values.
                double m = 0;
                double x = 0;
                double y = 0;
                if (i0 >= 0 && j0 >= 0) {
                    const Data<T> &d00 = table[k0 + i0];
                    const Data<T> &d10 = table[k1 + i0];
                    const Data<T> &d01 = table[k0 + i1];
                    m += d00.m - (d10.m + d01.m);
                    x += d00.x - (d10.x + d01.x);
                    y += d00.y - (d10.y + d01.y);
                } else if (i0 >= 0) {
                    const Data<T> &d10 = table[k1 + i0];
                    m -= d10.m;
                    x -= d10.x;
                    y -= d10.y;
                } else if (j0 >= 0) {
                    const Data<T> &d01 = table[k0 + i1];
                    m -= d01.m;
                    x -= d01.x;
                    y -= d01.y;
                }
                const Data<T> &d11 = table[k1 + i1];
                m += d11.m;
                x += d11.x;
                y += d11.y;

                // Compute the thresholds
                dst[k] = false;
                if (mask[k] && m >= min_count_ && x >= 0 && src[k] > threshold_) {
                    double a = m * y - x * x - x * (m - 1);
                    double b = m * src[k] - x;
                    double c = x * nsig_b_ * std::sqrt(2 * (m - 1));
                    double d = nsig_s_ * std::sqrt(x * m);
                    dst[k] = a > c && b > d;
                }
            }
        }
        return omp_get_wtime() - t0;
    }


    template <typename T>
    double compute_threshold_new(af::ref<Data<T>> table,
                           const af::const_ref<T, af::c_grid<2>> &src,
                           const af::const_ref<bool, af::c_grid<2>> &mask,
                           af::ref<bool, af::c_grid<2>> dst) {
        double t0 = omp_get_wtime();
        // Get the size of the image
        // I HAVE SWAPPED THESE TO MATCH THE DATA INDICES
        std::size_t ysize = src.accessor()[1];
        std::size_t xsize = src.accessor()[0];
        // ysize = 512;
        // xsize = 1028;
        // The kernel size
        int kxsize = kernel_size_[1];
        int kysize = kernel_size_[0];
        int i_offset0 = 0;
        int i_offset1 = 1 * (E2XE_MOD_FAST+E2XE_GAP_FAST);
        int i_offset2 = 2 * (E2XE_MOD_FAST+E2XE_GAP_FAST);
        int i_offset3 = 3 * (E2XE_MOD_FAST+E2XE_GAP_FAST);

        for (size_t module_row_num=0; module_row_num<8; module_row_num++) {
            int j_offset = module_row_num * (E2XE_MOD_SLOW + E2XE_GAP_SLOW);
        // Calculate the local mean at every point
            for (std::size_t j = 0; j < 512; ++j) {
                int j_im = j_offset + j;
                int j0 = j_im - kysize - 1;
                int j1 = j_im + kysize;
                j1 = j1 < (j_offset + E2XE_MOD_SLOW) ? j1 : (j_offset + E2XE_MOD_SLOW) - 1;
                int k0 = j0 * E2XE_16M_FAST;
                int k1 = j1 * E2XE_16M_FAST;
                int k_0 = (j_offset + j) * E2XE_16M_FAST;
                int k_1 = k_0 + E2XE_MOD_FAST + E2XE_GAP_FAST;
                int k_2 = k_1 + E2XE_MOD_FAST + E2XE_GAP_FAST;
                int k_3 = k_2 + E2XE_MOD_FAST + E2XE_GAP_FAST;
                for (std::size_t i = 0; i < 1028; ++i, ++k_0, ++k_1, ++k_2, ++k_3) {

                    int i_im_0 = i_offset0 + i;
                    int i_im_1 = i_offset1 + i;
                    int i_im_2 = i_offset2 + i;
                    int i_im_3 = i_offset3 + i;

                    int i0_0 = i_im_0 - kxsize - 1;
                    int i0_1 = i_im_1 - kxsize - 1;
                    int i0_2 = i_im_2 - kxsize - 1;
                    int i0_3 = i_im_3 - kxsize - 1;

                    int i1_0 = i_im_0 + kxsize;
                    int i1_1 = i_im_1 + kxsize;
                    int i1_2 = i_im_2 + kxsize;
                    int i1_3 = i_im_3 + kxsize;

                    int mod_right0 = i_offset0 + E2XE_MOD_FAST;
                    int mod_right1 = i_offset1 + E2XE_MOD_FAST;
                    int mod_right2 = i_offset2 + E2XE_MOD_FAST;
                    int mod_right3 = i_offset3 + E2XE_MOD_FAST;

                    i1_0 = i1_0 < mod_right0 ? i1_0 : mod_right0 - 1;
                    i1_1 = i1_1 < mod_right1 ? i1_1 : mod_right1 - 1;
                    i1_2 = i1_2 < mod_right2 ? i1_2 : mod_right2 - 1;
                    i1_3 = i1_3 < mod_right3 ? i1_3 : mod_right3 - 1;

                    double m0 = 0;
                    double x0 = 0;
                    double y0 = 0;
                    double m1 = 0;
                    double x1 = 0;
                    double y1 = 0;
                    double m2 = 0;
                    double x2 = 0;
                    double y2 = 0;
                    double m3 = 0;
                    double x3 = 0;
                    double y3 = 0;

                    if (i0_0 >= i_offset0 && j0 >= j_offset) {
                        const Data<T> &d00_0 = table[k0 + i0_0];
                        const Data<T> &d10_0 = table[k1 + i0_0];
                        const Data<T> &d01_0 = table[k0 + i1_0];
                        m0 += d00_0.m - (d10_0.m + d01_0.m);
                        x0 += d00_0.x - (d10_0.x + d01_0.x);
                        y0 += d00_0.y - (d10_0.y + d01_0.y);

                        const Data<T> &d00_1 = table[k0 + i0_1];
                        const Data<T> &d10_1 = table[k1 + i0_1];
                        const Data<T> &d01_1 = table[k0 + i1_1];
                        m1 += d00_1.m - (d10_1.m + d01_1.m);
                        x1 += d00_1.x - (d10_1.x + d01_1.x);
                        y1 += d00_1.y - (d10_1.y + d01_1.y);

                        const Data<T> &d00_2 = table[k0 + i0_2];
                        const Data<T> &d10_2 = table[k1 + i0_2];
                        const Data<T> &d01_2 = table[k0 + i1_2];
                        m2 += d00_2.m - (d10_2.m + d01_2.m);
                        x2 += d00_2.x - (d10_2.x + d01_2.x);
                        y2 += d00_2.y - (d10_2.y + d01_2.y);

                        const Data<T> &d00_3 = table[k0 + i0_3];
                        const Data<T> &d10_3 = table[k1 + i0_3];
                        const Data<T> &d01_3 = table[k0 + i1_3];
                        m3 += d00_3.m - (d10_3.m + d01_3.m);
                        x3 += d00_3.x - (d10_3.x + d01_3.x);
                        y3 += d00_3.y - (d10_3.y + d01_3.y);
                    } else if (i0_0 >= i_offset0) {
                        const Data<T> &d10_0 = table[k1 + i0_0];
                        m0 -= d10_0.m;
                        x0 -= d10_0.x;
                        y0 -= d10_0.y;

                        const Data<T> &d10_1 = table[k1 + i0_1];
                        m1 -= d10_1.m;
                        x1 -= d10_1.x;
                        y1 -= d10_1.y;

                        const Data<T> &d10_2 = table[k1 + i0_2];
                        m2 -= d10_2.m;
                        x2 -= d10_2.x;
                        y2 -= d10_2.y;

                        const Data<T> &d10_3 = table[k1 + i0_3];
                        m3 -= d10_3.m;
                        x3 -= d10_3.x;
                        y3 -= d10_3.y;
                    } else if (j0 >= j_offset) {
                        const Data<T> &d01_0 = table[k0 + i1_0];
                        m0 -= d01_0.m;
                        x0 -= d01_0.x;
                        y0 -= d01_0.y;

                        const Data<T> &d01_1 = table[k0 + i1_1];
                        m1 -= d01_1.m;
                        x1 -= d01_1.x;
                        y1 -= d01_1.y;

                        const Data<T> &d01_2 = table[k0 + i1_2];
                        m2 -= d01_2.m;
                        x2 -= d01_2.x;
                        y2 -= d01_2.y;

                        const Data<T> &d01_3 = table[k0 + i1_3];
                        m3 -= d01_3.m;
                        x3 -= d01_3.x;
                        y3 -= d01_3.y;
                    }

                    const Data<T> &d11_0 = table[k1 + i1_0];
                    m0 += d11_0.m;
                    x0 += d11_0.x;
                    y0 += d11_0.y;

                    const Data<T> &d11_1 = table[k1 + i1_1];
                    m1 += d11_1.m;
                    x1 += d11_1.x;
                    y1 += d11_1.y;

                    const Data<T> &d11_2 = table[k1 + i1_2];
                    m2 += d11_2.m;
                    x2 += d11_2.x;
                    y2 += d11_2.y;

                    const Data<T> &d11_3 = table[k1 + i1_3];
                    m3 += d11_3.m;
                    x3 += d11_3.x;
                    y3 += d11_3.y;

                    if (mask[k_0] && m0 >= min_count_ && x0 >= 0 && src[k_0] > threshold_) {
                        double a = m0 * y0 - x0 * x0 - x0 * (m0 - 1);
                        double b = m0 * src[k_0] - x0;
                        double c = x0 * nsig_b_ * std::sqrt(2 * (m0 - 1));
                        double d = nsig_s_ * std::sqrt(x0 * m0);
                        dst[k_0] = a > c && b > d;
                    }

                    if (mask[k_1] && m1 >= min_count_ && x1 >= 0 && src[k_1] > threshold_) {
                        double a = m1 * y1 - x1 * x1 - x1 * (m1 - 1);
                        double b = m1 * src[k_1] - x1;
                        double c = x1 * nsig_b_ * std::sqrt(2 * (m1 - 1));
                        double d = nsig_s_ * std::sqrt(x1 * m1);
                        dst[k_1] = a > c && b > d;
                    }

                    if (mask[k_2] && m2 >= min_count_ && x2 >= 0 && src[k_2] > threshold_) {
                        double a = m2 * y2 - x2 * x2 - x2 * (m2 - 1);
                        double b = m2 * src[k_2] - x2;
                        double c = x2 * nsig_b_ * std::sqrt(2 * (m2 - 1));
                        double d = nsig_s_ * std::sqrt(x2 * m2);
                        dst[k_2] = a > c && b > d;
                    }

                    if (mask[k_3] && m3 >= min_count_ && x3 >= 0 && src[k_3] > threshold_) {
                        double a = m3 * y3 - x3 * x3 - x3 * (m3 - 1);
                        double b = m3 * src[k_3] - x3;
                        double c = x3 * nsig_b_ * std::sqrt(2 * (m3 - 1));
                        double d = nsig_s_ * std::sqrt(x3 * m3);
                        dst[k_3] = a > c && b > d;
                    }
                }
            }
        }
        return omp_get_wtime() - t0;
    }


    /**
     * Compute the threshold
     * @param src - The input array
     * @param mask - The mask array
     * @param gain - The gain array
     * @param dst The output array
     */
    template <typename T>
    void compute_threshold(af::ref<Data<T>> table,
                           const af::const_ref<T, af::c_grid<2>> &src,
                           const af::const_ref<bool, af::c_grid<2>> &mask,
                           const af::const_ref<double, af::c_grid<2>> &gain,
                           af::ref<bool, af::c_grid<2>> dst) {
        // Get the size of the image
        std::size_t ysize = src.accessor()[0];
        std::size_t xsize = src.accessor()[1];

        // The kernel size
        int kxsize = kernel_size_[1];
        int kysize = kernel_size_[0];

        // Calculate the local mean at every point
        for (std::size_t j = 0, k = 0; j < ysize; ++j) {
            for (std::size_t i = 0; i < xsize; ++i, ++k) {
                int i0 = i - kxsize - 1, i1 = i + kxsize;
                int j0 = j - kysize - 1, j1 = j + kysize;
                i1 = i1 < xsize ? i1 : xsize - 1;
                j1 = j1 < ysize ? j1 : ysize - 1;
                int k0 = j0 * xsize;
                int k1 = j1 * xsize;

                // Compute the number of points valid in the local area,
                // the sum of the pixel values and the num of the squared pixel
                // values.
                double m = 0;
                double x = 0;
                double y = 0;
                if (i0 >= 0 && j0 >= 0) {
                    const Data<T> &d00 = table[k0 + i0];
                    const Data<T> &d10 = table[k1 + i0];
                    const Data<T> &d01 = table[k0 + i1];
                    m += d00.m - (d10.m + d01.m);
                    x += d00.x - (d10.x + d01.x);
                    y += d00.y - (d10.y + d01.y);
                } else if (i0 >= 0) {
                    const Data<T> &d10 = table[k1 + i0];
                    m -= d10.m;
                    x -= d10.x;
                    y -= d10.y;
                } else if (j0 >= 0) {
                    const Data<T> &d01 = table[k0 + i1];
                    m -= d01.m;
                    x -= d01.x;
                    y -= d01.y;
                }
                const Data<T> &d11 = table[k1 + i1];
                m += d11.m;
                x += d11.x;
                y += d11.y;

                // Compute the thresholds
                dst[k] = false;
                if (mask[k] && m >= min_count_ && x >= 0 && src[k] > threshold_) {
                    double a = m * y - x * x;
                    double b = m * src[k] - x;
                    double c = gain[k] * x * (m - 1 + nsig_b_ * std::sqrt(2 * (m - 1)));
                    double d = nsig_s_ * std::sqrt(gain[k] * x * m);
                    dst[k] = a > c && b > d;
                }
            }
        }
    }

    /**
     * Compute the threshold for the given image and mask.
     * @param src - The input image array.
     * @param mask - The mask array.
     * @param dst - The destination array.
     */
    template <typename T>
    void threshold(const af::const_ref<T, af::c_grid<2>> &src,
                   const af::const_ref<bool, af::c_grid<2>> &mask,
                   af::ref<bool, af::c_grid<2>> dst) {
        // check the input
        DIALS_ASSERT(src.accessor().all_eq(image_size_));
        DIALS_ASSERT(src.accessor().all_eq(mask.accessor()));
        DIALS_ASSERT(src.accessor().all_eq(dst.accessor()));

        // Get the table
        DIALS_ASSERT(sizeof(T) <= sizeof(double));

        // Cast the buffer to the table type
        af::ref<Data<T>> table(reinterpret_cast<Data<T> *>(&buffer_[0]),
                               buffer_.size());

        // compute the summed area table
        double old_sat_time = compute_sat(table, src, mask);

        // Compute the image threshold
        double old_thresh_time = compute_threshold(table, src, mask, dst);
    }

    /**
     * Compute the threshold for the given image and mask.
     * @param src - The input image array.
     * @param mask - The mask array.
     * @param gain - The gain array
     * @param dst - The destination array.
     */
    template <typename T>
    void threshold_w_gain(const af::const_ref<T, af::c_grid<2>> &src,
                          const af::const_ref<bool, af::c_grid<2>> &mask,
                          const af::const_ref<double, af::c_grid<2>> &gain,
                          af::ref<bool, af::c_grid<2>> dst) {
        // check the input
        DIALS_ASSERT(src.accessor().all_eq(image_size_));
        DIALS_ASSERT(src.accessor().all_eq(mask.accessor()));
        DIALS_ASSERT(src.accessor().all_eq(gain.accessor()));
        DIALS_ASSERT(src.accessor().all_eq(dst.accessor()));

        // Get the table
        DIALS_ASSERT(sizeof(T) <= sizeof(double));

        // Cast the buffer to the table type
        af::ref<Data<T>> table((Data<T> *)&buffer_[0], buffer_.size());

        // compute the summed area table
        compute_sat(table, src, mask);

        // Compute the image threshold
        compute_threshold(table, src, mask, gain, dst);
    }

  private:
    int2 image_size_;
    int2 kernel_size_;
    double nsig_b_;
    double nsig_s_;
    double threshold_;
    int min_count_;
    std::vector<char> buffer_;
};

class DispersionThresholdModules {
  public:
    /**
     * Enable more efficient memory usage by putting components required for the
     * summed area table closer together in memory
     */
    template <typename T>
    struct Data {
        int m;
        T x;
        T y;
    };

    DispersionThresholdModules(int2 image_size,
                        int2 kernel_size,
                        double nsig_b,
                        double nsig_s,
                        double threshold,
                        int min_count)
        : image_size_(image_size),
          kernel_size_(kernel_size),
          nsig_b_(nsig_b),
          nsig_s_(nsig_s),
          threshold_(threshold),
          min_count_(min_count) {
        // Check the input
        DIALS_ASSERT(threshold_ >= 0);
        DIALS_ASSERT(nsig_b >= 0 && nsig_s >= 0);
        DIALS_ASSERT(image_size.all_gt(0));
        DIALS_ASSERT(kernel_size.all_gt(0));

        // Ensure the min counts are valid
        std::size_t num_kernel = (2 * kernel_size[0] + 1) * (2 * kernel_size[1] + 1);
        if (min_count_ <= 0) {
            min_count_ = num_kernel;
        } else {
            DIALS_ASSERT(min_count_ <= num_kernel && min_count_ > 1);
        }

        // Allocate the buffer
        std::size_t element_size = sizeof(Data<double>);
        buffer_.resize(element_size * image_size[0] * image_size[1]);
    }

    /**
     * Compute the summed area tables for the mask, src and src^2 for the module_num-th module.
     * @param src The input array
     * @param mask The mask array
     */
    template <typename T>
    double compute_module_sat(af::ref<Data<T>> table,
                     const af::const_ref<T, af::c_grid<2>> &src,
                     const af::const_ref<bool, af::c_grid<2>> &mask,
                     int module_num) {
        double t0 = omp_get_wtime();
        // Largest value to consider
        const T BIG = (1 << 24);  // About 16m counts

        // Get the size of the image
        std::size_t ysize = E2XE_MOD_SLOW;
        std::size_t xsize = E2XE_MOD_FAST;

        std::size_t offset = (module_num / E2XE_16M_NFAST) * (E2XE_MOD_SLOW + E2XE_GAP_SLOW) * E2XE_16M_FAST + (module_num % E2XE_16M_NFAST) * (E2XE_MOD_FAST + E2XE_GAP_FAST);
        std::size_t row_step = E2XE_16M_FAST-E2XE_MOD_FAST;
        // Create the summed area table
        for (std::size_t j = 0, k = offset; j < ysize; ++j, k+=row_step) {
            int m = 0;
            T x = 0;
            T y = 0;
            for (std::size_t i = 0; i < xsize; ++i, ++k) {
                int mm = (mask[k] && src[k] < BIG) ? 1 : 0;
                m += mm;
                x += mm * src[k];
                y += mm * src[k] * src[k];
                if (j == 0) {
                    table[k].m = m;
                    table[k].x = x;
                    table[k].y = y;
                } else {
                    table[k].m = table[k - E2XE_16M_FAST].m + m;
                    table[k].x = table[k - E2XE_16M_FAST].x + x;
                    table[k].y = table[k - E2XE_16M_FAST].y + y;
                }
            }
        }
        
        return omp_get_wtime() - t0;
    }

    /**
     * Compute the threshold for a particular module
     * @param src - The input array
     * @param mask - The mask array
     * @param dst The output array
     * @param module_num - The index of the module
     */
    template <typename T>
    double compute_module_threshold(af::ref<Data<T>> table,
                           const af::const_ref<T, af::c_grid<2>> &src,
                           const af::const_ref<bool, af::c_grid<2>> &mask,
                           af::ref<bool, af::c_grid<2>> dst,
                           int module_num) {
        double t0 = omp_get_wtime();
        // Get the size of the image
        std::size_t ysize = E2XE_MOD_SLOW;
        std::size_t xsize = E2XE_MOD_FAST;
    
        std::size_t offset = (module_num / E2XE_16M_NFAST) * (E2XE_MOD_SLOW + E2XE_GAP_SLOW) * E2XE_16M_FAST + (module_num % E2XE_16M_NFAST) * (E2XE_MOD_FAST + E2XE_GAP_FAST);
        std::size_t row_step = E2XE_16M_FAST-E2XE_MOD_FAST;
        int i_offset = (module_num % E2XE_16M_NFAST) * (E2XE_MOD_FAST + E2XE_GAP_FAST);
        int j_offset = (module_num / E2XE_16M_NFAST) * (E2XE_MOD_SLOW + E2XE_GAP_SLOW);

        // The kernel size
        int kxsize = kernel_size_[1];
        int kysize = kernel_size_[0];

        // Calculate the local mean at every point
        for (std::size_t j = 0, k=offset; j < ysize; ++j, k+=row_step) {
            for (std::size_t i = 0; i < xsize; ++i, ++k) {

                dst[k] = false;

                // Full image i=x, j=y
                int i_im = i_offset + i;
                int j_im = j_offset + j;

                // Full image coordinates
                int i0 = i_im - kxsize - 1;
                int i1 = i_im + kxsize;
                int j0 = j_im - kysize - 1;
                int j1 = j_im + kysize;

                // j0,i0,j1,i1 are in image-space
                // but - we still need to cut off the kernel edges
                // at the edge of the module, not the edge of the
                // image
                int mod_right = i_offset + E2XE_MOD_FAST;
                int mod_bottom = j_offset + E2XE_MOD_SLOW;

                i1 = i1 < mod_right ? i1 : mod_right - 1;
                j1 = j1 < mod_bottom ? j1 : mod_bottom - 1;

                // Array index of the first pixel in the first and last
                // row of the kernel
                int k0 = j0 * E2XE_16M_FAST;
                int k1 = j1 * E2XE_16M_FAST;

                // Compute the number of points valid in the local area,
                // the sum of the pixel values and the sum of the squared pixel
                // values.
                double m = 0;
                double x = 0;
                double y = 0;
                if (i0 >= i_offset && j0 >= j_offset) {
                    const Data<T> &d00 = table[k0 + i0];
                    const Data<T> &d10 = table[k1 + i0];
                    const Data<T> &d01 = table[k0 + i1];
                    m += d00.m - (d10.m + d01.m);
                    x += d00.x - (d10.x + d01.x);
                    y += d00.y - (d10.y + d01.y);
                } else if (i0 >= i_offset) {
                    const Data<T> &d10 = table[k1 + i0];
                    m -= d10.m;
                    x -= d10.x;
                    y -= d10.y;
                } else if (j0 >= j_offset) {
                    const Data<T> &d01 = table[k0 + i1];
                    m -= d01.m;
                    x -= d01.x;
                    y -= d01.y;
                }
                const Data<T> &d11 = table[k1 + i1];
                m += d11.m;
                x += d11.x;
                y += d11.y;


                // Compute the thresholds
                // dst[k] = false;
                if (mask[k] && m >= min_count_ && x >= 0 && src[k] > threshold_) {
                    double a = m * y - x * x - x * (m - 1);
                    double b = m * src[k] - x;
                    double c = x * nsig_b_ * std::sqrt(2 * (m - 1));
                    double d = nsig_s_ * std::sqrt(x * m);
                    dst[k] = a > c && b > d;
                }
            }
        }
        return omp_get_wtime() - t0;
    }

    /**
     * Compute the threshold for the given image and mask.
     * @param src - The input image array.
     * @param mask - The mask array.
     * @param dst - The destination array.
     */
    template <typename T>
    void threshold(const af::const_ref<T, af::c_grid<2>> &src,
                   const af::const_ref<bool, af::c_grid<2>> &mask,
                   af::ref<bool, af::c_grid<2>> dst) {
        // check the input
        DIALS_ASSERT(src.accessor().all_eq(image_size_));
        DIALS_ASSERT(src.accessor().all_eq(mask.accessor()));
        DIALS_ASSERT(src.accessor().all_eq(dst.accessor()));

        // Get the table
        DIALS_ASSERT(sizeof(T) <= sizeof(double));

        // Cast the buffer to the table type
        af::ref<Data<T>> table(reinterpret_cast<Data<T> *>(&buffer_[0]),
                               buffer_.size());

        int n_modules = E2XE_16M_NSLOW * E2XE_16M_NFAST;

        for (size_t k=0; k<E2XE_16M_FAST*E2XE_16M_SLOW; ++k) dst[k] = false;
    #pragma omp parallel for default(none) shared(n_modules, table, src, mask, dst) num_threads(omp_get_max_threads()/2)
        for (size_t n=0; n<n_modules; n++) {
            compute_module_sat(table, src, mask, n);
            compute_module_threshold(table, src, mask, dst, n);
        }
    }

  private:
    int2 image_size_;
    int2 kernel_size_;
    double nsig_b_;
    double nsig_s_;
    double threshold_;
    int min_count_;
    std::vector<char> buffer_;
};

/**
 * A class to compute the threshold using index of dispersion
 */
class DispersionExtendedThreshold {
  public:
    /**
     * Enable more efficient memory usage by putting components required for the
     * summed area table closer together in memory
     */
    template <typename T>
    struct Data {
        int m;
        T x;
        T y;
    };

    DispersionExtendedThreshold(int2 image_size,
                                int2 kernel_size,
                                double nsig_b,
                                double nsig_s,
                                double threshold,
                                int min_count)
        : image_size_(image_size),
          kernel_size_(kernel_size),
          nsig_b_(nsig_b),
          nsig_s_(nsig_s),
          threshold_(threshold),
          min_count_(min_count) {
        // Check the input
        DIALS_ASSERT(threshold_ >= 0);
        DIALS_ASSERT(nsig_b >= 0 && nsig_s >= 0);
        DIALS_ASSERT(image_size.all_gt(0));
        DIALS_ASSERT(kernel_size.all_gt(0));

        // Ensure the min counts are valid
        std::size_t num_kernel = (2 * kernel_size[0] + 1) * (2 * kernel_size[1] + 1);
        if (min_count_ <= 0) {
            min_count_ = num_kernel;
        } else {
            DIALS_ASSERT(min_count_ <= num_kernel && min_count_ > 1);
        }

        // Allocate the buffer
        std::size_t element_size = sizeof(Data<double>);
        buffer_.resize(element_size * image_size[0] * image_size[1]);
    }

    /**
     * Compute the summed area tables for the mask, src and src^2.
     * @param src The input array
     * @param mask The mask array
     */
    template <typename T>
    void compute_sat(af::ref<Data<T>> table,
                     const af::const_ref<T, af::c_grid<2>> &src,
                     const af::const_ref<bool, af::c_grid<2>> &mask) {
        // Largest value to consider
        const T BIG = (1 << 24);  // About 16m counts

        // Get the size of the image
        std::size_t ysize = src.accessor()[0];
        std::size_t xsize = src.accessor()[1];

        // Create the summed area table
        for (std::size_t j = 0, k = 0; j < ysize; ++j) {
            int m = 0;
            T x = 0;
            T y = 0;
            for (std::size_t i = 0; i < xsize; ++i, ++k) {
                int mm = (mask[k] && src[k] < BIG) ? 1 : 0;
                m += mm;
                x += mm * src[k];
                y += mm * src[k] * src[k];
                if (j == 0) {
                    table[k].m = m;
                    table[k].x = x;
                    table[k].y = y;
                } else {
                    table[k].m = table[k - xsize].m + m;
                    table[k].x = table[k - xsize].x + x;
                    table[k].y = table[k - xsize].y + y;
                }
            }
        }
    }

    /**
     * Compute the threshold
     * @param src - The input array
     * @param mask - The mask array
     * @param dst The output array
     */
    template <typename T>
    void compute_dispersion_threshold(af::ref<Data<T>> table,
                                      const af::const_ref<T, af::c_grid<2>> &src,
                                      const af::const_ref<bool, af::c_grid<2>> &mask,
                                      af::ref<bool, af::c_grid<2>> dst) {
        // Get the size of the image
        std::size_t ysize = src.accessor()[0];
        std::size_t xsize = src.accessor()[1];

        // The kernel size
        int kxsize = kernel_size_[1];
        int kysize = kernel_size_[0];

        // Calculate the local mean at every point
        for (std::size_t j = 0, k = 0; j < ysize; ++j) {
            for (std::size_t i = 0; i < xsize; ++i, ++k) {
                int i0 = i - kxsize - 1, i1 = i + kxsize;
                int j0 = j - kysize - 1, j1 = j + kysize;
                i1 = i1 < xsize ? i1 : xsize - 1;
                j1 = j1 < ysize ? j1 : ysize - 1;
                int k0 = j0 * xsize;
                int k1 = j1 * xsize;

                // Compute the number of points valid in the local area,
                // the sum of the pixel values and the sum of the squared pixel
                // values.
                double m = 0;
                double x = 0;
                double y = 0;
                if (i0 >= 0 && j0 >= 0) {
                    const Data<T> &d00 = table[k0 + i0];
                    const Data<T> &d10 = table[k1 + i0];
                    const Data<T> &d01 = table[k0 + i1];
                    m += d00.m - (d10.m + d01.m);
                    x += d00.x - (d10.x + d01.x);
                    y += d00.y - (d10.y + d01.y);
                } else if (i0 >= 0) {
                    const Data<T> &d10 = table[k1 + i0];
                    m -= d10.m;
                    x -= d10.x;
                    y -= d10.y;
                } else if (j0 >= 0) {
                    const Data<T> &d01 = table[k0 + i1];
                    m -= d01.m;
                    x -= d01.x;
                    y -= d01.y;
                }
                const Data<T> &d11 = table[k1 + i1];
                m += d11.m;
                x += d11.x;
                y += d11.y;

                // Compute the thresholds
                dst[k] = false;
                if (mask[k] && m >= min_count_ && x >= 0) {
                    double a = m * y - x * x - x * (m - 1);
                    double c = x * nsig_b_ * std::sqrt(2 * (m - 1));
                    dst[k] = (a > c);
                }
            }
        }
    }

    /**
     * Compute the threshold
     * @param src - The input array
     * @param mask - The mask array
     * @param gain - The gain array
     * @param dst The output array
     */
    template <typename T>
    void compute_dispersion_threshold(af::ref<Data<T>> table,
                                      const af::const_ref<T, af::c_grid<2>> &src,
                                      const af::const_ref<bool, af::c_grid<2>> &mask,
                                      const af::const_ref<double, af::c_grid<2>> &gain,
                                      af::ref<bool, af::c_grid<2>> dst) {
        // Get the size of the image
        std::size_t ysize = src.accessor()[0];
        std::size_t xsize = src.accessor()[1];

        // The kernel size
        int kxsize = kernel_size_[1];
        int kysize = kernel_size_[0];

        // Calculate the local mean at every point
        for (std::size_t j = 0, k = 0; j < ysize; ++j) {
            for (std::size_t i = 0; i < xsize; ++i, ++k) {
                int i0 = i - kxsize - 1, i1 = i + kxsize;
                int j0 = j - kysize - 1, j1 = j + kysize;
                i1 = i1 < xsize ? i1 : xsize - 1;
                j1 = j1 < ysize ? j1 : ysize - 1;
                int k0 = j0 * xsize;
                int k1 = j1 * xsize;

                // Compute the number of points valid in the local area,
                // the sum of the pixel values and the num of the squared pixel
                // values.
                double m = 0;
                double x = 0;
                double y = 0;
                if (i0 >= 0 && j0 >= 0) {
                    const Data<T> &d00 = table[k0 + i0];
                    const Data<T> &d10 = table[k1 + i0];
                    const Data<T> &d01 = table[k0 + i1];
                    m += d00.m - (d10.m + d01.m);
                    x += d00.x - (d10.x + d01.x);
                    y += d00.y - (d10.y + d01.y);
                } else if (i0 >= 0) {
                    const Data<T> &d10 = table[k1 + i0];
                    m -= d10.m;
                    x -= d10.x;
                    y -= d10.y;
                } else if (j0 >= 0) {
                    const Data<T> &d01 = table[k0 + i1];
                    m -= d01.m;
                    x -= d01.x;
                    y -= d01.y;
                }
                const Data<T> &d11 = table[k1 + i1];
                m += d11.m;
                x += d11.x;
                y += d11.y;

                // Compute the thresholds
                dst[k] = false;
                if (mask[k] && m >= min_count_ && x >= 0) {
                    double a = m * y - x * x;
                    double c = gain[k] * x * (m - 1 + nsig_b_ * std::sqrt(2 * (m - 1)));
                    dst[k] = (a > c);
                }
            }
        }
    }

    /**
     * Erode the dispersion mask
     * @param dst The dispersion mask
     */
    void erode_dispersion_mask(const af::const_ref<bool, af::c_grid<2>> &mask,
                               af::ref<bool, af::c_grid<2>> dst) {
        // The distance array
        af::versa<int, af::c_grid<2>> distance(dst.accessor(), 0);

        // Compute the chebyshev distance to the nearest valid background pixel
        chebyshev_distance(dst, false, distance.ref());

        // The erosion distance
        std::size_t erosion_distance = std::min(kernel_size_[0], kernel_size_[1]);

        // Compute the eroded mask
        for (std::size_t k = 0; k < dst.size(); ++k) {
            if (mask[k]) {
                dst[k] = !(dst[k] && distance[k] >= erosion_distance);
            } else {
                dst[k] = false;
            }
        }
    }

    /**
     * Compute the threshold
     * @param src - The input array
     * @param mask - The mask array
     * @param dst The output array
     */
    template <typename T>
    void compute_final_threshold(af::ref<Data<T>> table,
                                 const af::const_ref<T, af::c_grid<2>> &src,
                                 const af::const_ref<bool, af::c_grid<2>> &mask,
                                 af::ref<bool, af::c_grid<2>> dst) {
        // Get the size of the image
        std::size_t ysize = src.accessor()[0];
        std::size_t xsize = src.accessor()[1];

        // The kernel size
        int kxsize = kernel_size_[1] + 2;
        int kysize = kernel_size_[0] + 2;

        // Calculate the local mean at every point
        for (std::size_t j = 0, k = 0; j < ysize; ++j) {
            for (std::size_t i = 0; i < xsize; ++i, ++k) {
                int i0 = i - kxsize - 1, i1 = i + kxsize;
                int j0 = j - kysize - 1, j1 = j + kysize;
                i1 = i1 < xsize ? i1 : xsize - 1;
                j1 = j1 < ysize ? j1 : ysize - 1;
                int k0 = j0 * xsize;
                int k1 = j1 * xsize;

                // Compute the number of points valid in the local area,
                // the sum of the pixel values and the sum of the squared pixel
                // values.
                double m = 0;
                double x = 0;
                if (i0 >= 0 && j0 >= 0) {
                    const Data<T> &d00 = table[k0 + i0];
                    const Data<T> &d10 = table[k1 + i0];
                    const Data<T> &d01 = table[k0 + i1];
                    m += d00.m - (d10.m + d01.m);
                    x += d00.x - (d10.x + d01.x);
                } else if (i0 >= 0) {
                    const Data<T> &d10 = table[k1 + i0];
                    m -= d10.m;
                    x -= d10.x;
                } else if (j0 >= 0) {
                    const Data<T> &d01 = table[k0 + i1];
                    m -= d01.m;
                    x -= d01.x;
                }
                const Data<T> &d11 = table[k1 + i1];
                m += d11.m;
                x += d11.x;

                // Compute the thresholds. The pixel is marked True if:
                // 1. The pixel is valid
                // 2. It has 1 or more unmasked neighbours
                // 3. It is within the dispersion masked region
                // 4. It is greater than the global threshold
                // 5. It is greater than the local mean threshold
                //
                // Otherwise it is false
                if (mask[k] && m >= 0 && x >= 0) {
                    bool dispersion_mask = !dst[k];
                    bool global_mask = src[k] > threshold_;
                    double mean = (m >= 2 ? (x / m) : 0);
                    bool local_mask = src[k] >= (mean + nsig_s_ * std::sqrt(mean));
                    dst[k] = dispersion_mask && global_mask && local_mask;
                } else {
                    dst[k] = false;
                }
            }
        }
    }

    /**
     * Compute the threshold
     * @param src - The input array
     * @param mask - The mask array
     * @param dst The output array
     */
    template <typename T>
    void compute_final_threshold(af::ref<Data<T>> table,
                                 const af::const_ref<T, af::c_grid<2>> &src,
                                 const af::const_ref<bool, af::c_grid<2>> &mask,
                                 const af::const_ref<double, af::c_grid<2>> &gain,
                                 af::ref<bool, af::c_grid<2>> dst) {
        // Get the size of the image
        std::size_t ysize = src.accessor()[0];
        std::size_t xsize = src.accessor()[1];

        // The kernel size
        int kxsize = kernel_size_[1] + 2;
        int kysize = kernel_size_[0] + 2;

        // Calculate the local mean at every point
        for (std::size_t j = 0, k = 0; j < ysize; ++j) {
            for (std::size_t i = 0; i < xsize; ++i, ++k) {
                int i0 = i - kxsize - 1, i1 = i + kxsize;
                int j0 = j - kysize - 1, j1 = j + kysize;
                i1 = i1 < xsize ? i1 : xsize - 1;
                j1 = j1 < ysize ? j1 : ysize - 1;
                int k0 = j0 * xsize;
                int k1 = j1 * xsize;

                // Compute the number of points valid in the local area,
                // the sum of the pixel values and the sum of the squared pixel
                // values.
                double m = 0;
                double x = 0;
                if (i0 >= 0 && j0 >= 0) {
                    const Data<T> &d00 = table[k0 + i0];
                    const Data<T> &d10 = table[k1 + i0];
                    const Data<T> &d01 = table[k0 + i1];
                    m += d00.m - (d10.m + d01.m);
                    x += d00.x - (d10.x + d01.x);
                } else if (i0 >= 0) {
                    const Data<T> &d10 = table[k1 + i0];
                    m -= d10.m;
                    x -= d10.x;
                } else if (j0 >= 0) {
                    const Data<T> &d01 = table[k0 + i1];
                    m -= d01.m;
                    x -= d01.x;
                }
                const Data<T> &d11 = table[k1 + i1];
                m += d11.m;
                x += d11.x;

                // Compute the thresholds. The pixel is marked True if:
                // 1. The pixel is valid
                // 2. It has 1 or more unmasked neighbours
                // 3. It is within the dispersion masked region
                // 4. It is greater than the global threshold
                // 5. It is greater than the local mean threshold
                //
                // Otherwise it is false
                if (mask[k] && m >= 0 && x >= 0) {
                    bool dispersion_mask = !dst[k];
                    bool global_mask = src[k] > threshold_;
                    double mean = (m >= 2 ? (x / m) : 0);
                    bool local_mask =
                      src[k] >= (mean + nsig_s_ * std::sqrt(gain[k] * mean));
                    dst[k] = dispersion_mask && global_mask && local_mask;
                } else {
                    dst[k] = false;
                }
            }
        }
    }

    /**
     * Compute the threshold for the given image and mask.
     * @param src - The input image array.
     * @param mask - The mask array.
     * @param dst - The destination array.
     */
    template <typename T>
    void threshold(const af::const_ref<T, af::c_grid<2>> &src,
                   const af::const_ref<bool, af::c_grid<2>> &mask,
                   af::ref<bool, af::c_grid<2>> dst) {
        // check the input
        DIALS_ASSERT(src.accessor().all_eq(image_size_));
        DIALS_ASSERT(src.accessor().all_eq(mask.accessor()));
        DIALS_ASSERT(src.accessor().all_eq(dst.accessor()));

        // Get the table
        DIALS_ASSERT(sizeof(T) <= sizeof(double));

        // Cast the buffer to the table type
        af::ref<Data<T>> table(reinterpret_cast<Data<T> *>(&buffer_[0]),
                               buffer_.size());

        // compute the summed area table
        compute_sat(table, src, mask);

        // Compute the dispersion threshold. This output is in dst which contains
        // a mask where 1 is valid background and 0 is invalid pixels and stuff
        // above the dispersion threshold
        compute_dispersion_threshold(table, src, mask, dst);

        // Erode the dispersion mask
        erode_dispersion_mask(mask, dst);

        // Compute the summed area table again now excluding the threshold pixels
        compute_sat(table, src, dst);

        // Compute the final threshold
        compute_final_threshold(table, src, mask, dst);
    }

    /**
     * Compute the threshold for the given image and mask.
     * @param src - The input image array.
     * @param mask - The mask array.
     * @param gain - The gain array
     * @param dst - The destination array.
     */
    template <typename T>
    void threshold_w_gain(const af::const_ref<T, af::c_grid<2>> &src,
                          const af::const_ref<bool, af::c_grid<2>> &mask,
                          const af::const_ref<double, af::c_grid<2>> &gain,
                          af::ref<bool, af::c_grid<2>> dst) {
        // check the input
        DIALS_ASSERT(src.accessor().all_eq(image_size_));
        DIALS_ASSERT(src.accessor().all_eq(mask.accessor()));
        DIALS_ASSERT(src.accessor().all_eq(gain.accessor()));
        DIALS_ASSERT(src.accessor().all_eq(dst.accessor()));

        // Get the table
        DIALS_ASSERT(sizeof(T) <= sizeof(double));

        // Cast the buffer to the table type
        af::ref<Data<T>> table((Data<T> *)&buffer_[0], buffer_.size());

        // compute the summed area table
        compute_sat(table, src, mask);

        // Compute the dispersion threshold. This output is in dst which contains
        // a mask where 1 is valid background and 0 is invalid pixels and stuff
        // above the dispersion threshold
        compute_dispersion_threshold(table, src, mask, gain, dst);

        // Erode the dispersion mask
        erode_dispersion_mask(mask, dst);

        // Compute the summed area table again now excluding the threshold pixels
        compute_sat(table, src, dst);

        // Compute the final threshold
        compute_final_threshold(table, src, mask, gain, dst);
    }

  private:
    int2 image_size_;
    int2 kernel_size_;
    double nsig_b_;
    double nsig_s_;
    double threshold_;
    int min_count_;
    std::vector<char> buffer_;
};

}  // namespace baseline

template <typename T, typename internal_T = T, typename algo_T = baseline::DispersionThreshold>
class _spotfind_context {
  public:
    af::ref<bool, af::c_grid<2>> dst;
    bool *_dest_store = nullptr;
    af::tiny<int, 2> size;

    af::ref<internal_T, af::c_grid<2>> src_converted;
    internal_T *_src_converted_store;

    algo_T algo;

    _spotfind_context(size_t width, size_t height)
        : size(width, height),
          algo(size, kernel_size_, nsig_b_, nsig_s_, threshold_, min_count_) {
        _dest_store = new bool[width * height];
        dst = af::ref<bool, af::c_grid<2>>(_dest_store, af::c_grid<2>(width, height));
        // Make a place to convert sources to the internal type
        _src_converted_store = new internal_T[width * height];
        src_converted = af::ref<internal_T, af::c_grid<2>>(
          _src_converted_store, af::c_grid<2>(width, height));
    }
    ~_spotfind_context() {
        delete[] _dest_store;
        delete[] _src_converted_store;
    }
    void threshold(const af::const_ref<internal_T, af::c_grid<2>> &src,
                   const af::const_ref<bool, af::c_grid<2>> &mask) {
        algo.threshold(src_converted, mask, dst);
    }
};

void *spotfinder_create(size_t width, size_t height) {
    return new _spotfind_context<image_t_type, double>(width, height);
}
void spotfinder_free(void *context) {
    delete reinterpret_cast<_spotfind_context<image_t_type, double> *>(context);
}

void *spotfinder_create_f(size_t width, size_t height) {
    return new _spotfind_context<image_t_type, float>(width, height);
}
void spotfinder_free_f(void *context) {
    delete reinterpret_cast<_spotfind_context<image_t_type, float> *>(context);
}

void *spotfinder_create_new(size_t width, size_t height) {
    return new _spotfind_context<image_t_type, double, baseline::DispersionThresholdModules>(width, height);
}
void spotfinder_free_new(void *context) {
    delete reinterpret_cast<_spotfind_context<image_t_type, double, baseline::DispersionThresholdModules> *>(context);
}

uint32_t spotfinder_standard_dispersion(void *context, image_t *image) {
    auto ctx = reinterpret_cast<_spotfind_context<image_t_type, double> *>(context);

    // mask needs to convert uint8_t to bool
    auto mask = af::const_ref<bool, af::c_grid<2>>(
      reinterpret_cast<bool *>(image->mask), af::c_grid<2>(ctx->size[0], ctx->size[1]));

    // Convert all items from the source image to
    for (int i = 0; i < (ctx->size[0] * ctx->size[1]); ++i) {
        ctx->src_converted[i] = image->data[i];
    }

    ctx->threshold(ctx->src_converted, mask);
    // Let's count the number of destination pixels for now
    uint32_t pixel_count = 0;
    for (int i = 0; i < (ctx->size[0] * ctx->size[1]); ++i) {
        pixel_count += ctx->dst[i];
    }
    return pixel_count;
}

uint32_t spotfinder_standard_dispersion_modules(void *context, image_modules_t *image_modules, size_t index) {
    auto ctx = reinterpret_cast<_spotfind_context<image_t_type, double> *>(context);

    size_t offset = index * ctx->size[0] * ctx->size[1];

    // mask needs to convert uint8_t to bool
    auto mask = af::const_ref<bool, af::c_grid<2>>(
      reinterpret_cast<bool *>(&(image_modules->mask[offset])), af::c_grid<2>(ctx->size[0], ctx->size[1])); 

    // Convert all items from the source image to double
    for (int i = 0; i < (ctx->size[0] * ctx->size[1]); ++i) {
        ctx->src_converted[i] = image_modules->data[offset+i];
    }

    ctx->threshold(ctx->src_converted, mask);

    // Let's count the number of destination pixels for now
    uint32_t pixel_count = 0;
    for (int i = 0; i < (ctx->size[0] * ctx->size[1]); ++i) {
        pixel_count += ctx->dst[i];
    }
    // printf("Pixel count: %d\n", pixel_count);
    return pixel_count;
}

uint32_t spotfinder_standard_dispersion_modules_f(void *context, image_modules_t *image_modules, size_t index) {
    auto ctx = reinterpret_cast<_spotfind_context<image_t_type, float> *>(context);

    size_t offset = index * ctx->size[0] * ctx->size[1];

    // mask needs to convert uint8_t to bool
    auto mask = af::const_ref<bool, af::c_grid<2>>(
      reinterpret_cast<bool *>(&(image_modules->mask[offset])), af::c_grid<2>(ctx->size[0], ctx->size[1])); 

    // Convert all items from the source image to double
    for (int i = 0; i < (ctx->size[0] * ctx->size[1]); ++i) {
        ctx->src_converted[i] = image_modules->data[offset+i];
    }

    ctx->threshold(ctx->src_converted, mask);
    uint32_t pixel_count = 0;
    for (int i = 0; i < (ctx->size[0] * ctx->size[1]); ++i) {
        pixel_count += ctx->dst[i];
    }
    return pixel_count;
}

uint32_t spotfinder_standard_dispersion_modules_new(void *context, image_t *image) {
    auto ctx = reinterpret_cast<_spotfind_context<image_t_type, double, baseline::DispersionThresholdModules> *>(context);

    // mask needs to convert uint8_t to bool
    auto mask = af::const_ref<bool, af::c_grid<2>>(
      reinterpret_cast<bool *>(image->mask), af::c_grid<2>(ctx->size[0], ctx->size[1])); 

    // Convert all items from the source image to double
    for (int i = 0; i < (ctx->size[0] * ctx->size[1]); ++i) {
        ctx->src_converted[i] = image->data[i];
    }

    ctx->threshold(ctx->src_converted, mask);

    // Let's count the number of destination pixels for now
    uint32_t pixel_count = 0;
    for (int i = 0; i < (ctx->size[0] * ctx->size[1]); ++i) {
        pixel_count += ctx->dst[i];
    }
    return pixel_count;
}
