
#include <fmt/core.h>
#include <inttypes.h>

#include <CL/sycl.hpp>
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>

#include "common.hpp"
#include "eiger2xe.h"
#include "h5read.h"
#include "kernel.hpp"
#include "tinytiffwriter.h"

using namespace sycl;

const sycl::stream& operator<<(const sycl::stream& os, const PipedPixelsArray& obj) {
    os << "[ ";
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        if (i > 0) {
            os << ", ";
        }
        os << setw(2) << obj[i];
    }
    os << " ]";
    return os;
}

/// Return the profiling event time, in milliseconds, for an event
double event_ms(const sycl::event& e) {
    return 1e-6
           * static_cast<double>(
             e.get_profiling_info<info::event_profiling::command_end>()
             - e.get_profiling_info<info::event_profiling::command_start>());
}

/// Calculate the GigaBytes per second given bytes, time
double GBps(size_t bytes, double ms) {
    return (static_cast<double>(bytes) / 1e9) / (ms / 1000.0);
}

/// Return value of event in terms of GigaBytes per second
double event_GBps(const sycl::event& e, size_t bytes) {
    const double ms = event_ms(e);
    return GBps(bytes, ms);
}

/// Draw a subset of the pixel values for a 2D image array
/// fast, slow, width, height - describe the bounding box to draw
/// data_width, data_height - describe the full data array size
template <typename T>
void draw_image_data(const T* data,
                     size_t fast,
                     size_t slow,
                     size_t width,
                     size_t height,
                     size_t data_width,
                     size_t data_height) {
    std::string format_type = "";
    if constexpr (std::is_integral<T>::value) {
        format_type = "d";
    } else {
        format_type = ".1f";
    }

    // Walk over the data and get various metadata for generation
    // Maximum value
    T accum = 0;
    // Maximum format width for each column
    std::vector<int> col_widths;
    for (int col = fast; col < fast + width; ++col) {
        size_t maxw = fmt::formatted_size("{}", col);
        for (int row = slow; row < min(slow + height, data_height); ++row) {
            auto val = data[col + data_width * row];
            auto fmt_spec = fmt::format("{{:{}}}", format_type);
            maxw = std::max(maxw, fmt::formatted_size(fmt_spec, val));
            accum = max(accum, val);
        }
        col_widths.push_back(maxw);
    }

    // Draw a row header
    fmt::print("x =       ");
    for (int i = 0; i < width; ++i) {
        auto x = i + fast;
        fmt::print("{:{}} ", x, col_widths[i]);
    }
    fmt::print("\n         ┌");
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < col_widths[i]; ++j) {
            fmt::print("─");
        }
        fmt::print("─");
    }
    fmt::print("┐\n");

    for (int y = slow; y < min(slow + height, data_height); ++y) {
        if (y == slow) {
            fmt::print("y = {:2d} │", y);
        } else {
            fmt::print("    {:2d} │", y);
        }
        for (int i = fast; i < fast + width; ++i) {
            // Calculate color
            // Black, 232->255, White
            // Range of 24 colors, not including white. Split into 25 bins, so
            // that we have a whole black top bin
            // float bin_scale = -25
            auto dat = data[i + data_width * y];
            int color = 255 - ((float)dat / (float)accum) * 24;
            if (color <= 231) color = 0;
            if (dat < 0) {
                color = 9;
            }

            if (dat == accum) {
                fmt::print("\033[0m\033[1m");
            } else {
                fmt::print("\033[38;5;{}m", color);
            }
            auto fmt_spec =
              fmt::format("{{:{}{}}} ", col_widths[i - fast], format_type);
            fmt::print(fmt_spec, dat);
            if (dat == accum) {
                fmt::print("\033[0m");
            }
        }
        fmt::print("\033[0m│\n");
    }
}
template <typename T>
void draw_image_data(
  const multi_ptr<T, access::address_space::ext_intel_global_host_space>& data,
  size_t fast,
  size_t slow,
  size_t width,
  size_t height,
  size_t data_width,
  size_t data_height) {
    draw_image_data(
      static_cast<T*>(data.get()), fast, slow, width, height, data_width, data_height);
}

void check_allocs() {}
/** Basic sanity check on allocations.
*
* So that if they fail, we don't get a SEGV later. This is basically
* mostly laziness, though has the advantage of not messing up every
* single allocation with validation. Maybe the allocations could be
* wrapped instead....
*/
template <typename T, typename... R>
void check_allocs(T arg, R... args) {
    if (arg == nullptr) {
        throw std::bad_alloc{};
    }
    check_allocs(args...);
}

FindSpotsDebugOutput::FindSpotsDebugOutput(sycl::queue Q) {
#ifdef DEBUG_IMAGES
    sum = sycl::malloc_host<decltype(sum)::element_type>(SLOW * FAST, Q);
    sumsq = sycl::malloc_host<decltype(sumsq)::element_type>(SLOW * FAST, Q);
    dispersion = sycl::malloc_host<float>(SLOW * FAST, Q);
    mean = sycl::malloc_host<float>(SLOW * FAST, Q);
    variance = sycl::malloc_host<float>(SLOW * FAST, Q);
    threshold = sycl::malloc_host<bool>(SLOW * FAST, Q);

    check_allocs(sum, sumsq, dispersion, mean, variance, threshold);
    for (std::size_t i = 0; i < SLOW * FAST; ++i) {
        sum[i] = sumsq[i] = 42;
        dispersion[i] = mean[i] = variance[i] = -1.0f;
        threshold[i] = false;
    }
    // memset(sum, 0, SLOW*FAST*sizeof(H5Read::image_type));
    // memset(sumsq, 0, SLOW*FAST*sizeof(H5Read::image_type));
    // memset(dispersion, 0, SLOW*FAST*sizeof(float));
    // memset(mean, 0, SLOW*FAST*sizeof(float));
    // memset(variance, 0, SLOW*FAST*sizeof(float));
    // memset(threshold, 0, SLOW*FAST*sizeof(bool));
#else
    sum = nullptr;
    sumsq = nullptr;
    dispersion = nullptr;
    mean = nullptr;
    variance = nullptr;
    threshold = nullptr;
#endif
}

/** Calculate a kernel sum, the simplest possible implementation.
 * 
 * This is **slow**, even from the perspective of something running
 * infrequently. It's relatively simple to get the algorithm correct,
 * however, so is useful for validating other algorithms.
 **/
template <typename Tin>
auto calculate_kernel_sum_slow(Tin* data, std::size_t fast, std::size_t slow) {
    auto out_d = std::make_unique<Tin[]>(slow * fast);
    Tin* out = out_d.get();
    for (int y = 0; y < slow; ++y) {
        int y0 = std::max(0, y - KERNEL_HEIGHT);
        int y1 = std::min((int)slow, y + KERNEL_HEIGHT + 1);
        // std::size_t y1 = std::min((int)slow-1, y + KERNEL_HEIGHT);
        for (int x = 0; x < fast; ++x) {
            int x0 = std::max(0, x - KERNEL_WIDTH);
            int x1 = std::min((int)fast, x + KERNEL_WIDTH + 1);
            Tin acc{};
            for (int ky = y0; ky < y1; ++ky) {
                for (int kx = x0; kx < x1; ++kx) {
                    acc += data[(ky * fast) + kx];
                }
            }
            out[(y * fast) + x] = acc;
            if (y == 2 && x == 12) {
                fmt::print("Got acc: {}\n", acc);
            }
        }
    }
    return out_d;
}

/** Calculate a kernel sum, on the host, using an SAT.
 * 
 * This is designed for non-offloading calculations for e.g. crosscheck
 * or precalculation (like the mask).
 **/
template <typename Tin, typename Taccumulator = Tin>
auto calculate_kernel_sum_sat(Tin* data, std::size_t fast, std::size_t slow)
  -> std::unique_ptr<Tin[]> {
    auto sat_d = std::make_unique<Taccumulator[]>(slow * fast);
    Taccumulator* sat = sat_d.get();
    for (int y = 0; y < slow; ++y) {
        Taccumulator acc = 0;
        for (int x = 0; x < fast; ++x) {
            acc += data[y * fast + x];
            if (y == 0) {
                sat[y * fast + x] = acc;
            } else {
                sat[y * fast + x] = acc + sat[(y - 1) * fast + x];
            }
        }
    }

    // Now evaluate the (fixed size) kernel across this SAT
    auto out_d = std::make_unique<Tin[]>(slow * fast);
    Tin* out = out_d.get();
    for (int y = 0; y < slow; ++y) {
        int y0 = y - KERNEL_HEIGHT - 1;
        int y1 = std::min((int)slow - 1, y + KERNEL_HEIGHT);
        for (int x = 0; x < fast; ++x) {
            int x0 = x - KERNEL_WIDTH - 1;
            int x1 = std::min((int)fast - 1, x + KERNEL_WIDTH);

            Taccumulator tl{}, tr{}, bl{};
            Taccumulator br = sat[y1 * fast + x1];

            if (y0 >= 0 && x0 >= 0) {
                // Top left fully inside kernel
                tl = sat[y0 * fast + x0];
                tr = sat[y0 * fast + x1];
                bl = sat[y1 * fast + x0];
            } else if (x0 >= 0) {
                // Top rows - y0 outside range
                bl = sat[y1 * fast + x0];
            } else if (y0 >= 0) {
                // Left rows - x0 outside range
                tr = sat[y0 * fast + x1];
            }
            out[y * fast + x] = br - (bl + tr) + tl;
        }
    }

    return out_d;
}

bool compare_results(const uint16_t* left,
                     const uint16_t* right,
                     std::size_t num_pixels) {
    for (std::size_t i = 0; i < num_pixels; ++i) {
        if (left[i] != right[i]) {
            return false;
        }
    }
    return true;
}

template <typename T>
auto write_tiff(std::string filename_, int number, T* data) {
    std::string type = "UInt";
    auto tif_type = TinyTIFFWriter_UInt;
    if constexpr (std::is_floating_point<T>::value) {
        tif_type = TinyTIFFWriter_Float;
        type = "Float";
    } else if constexpr (std::is_signed<T>::value) {
        tif_type = TinyTIFFWriter_Int;
        type = "Int";
    }
    auto filename = fmt::format(filename_, number);
    TinyTIFFWriterFile* tif = TinyTIFFWriter_open(filename.c_str(),
                                                  sizeof(T) * 8,
                                                  tif_type,
                                                  1,
                                                  FAST,
                                                  SLOW,
                                                  TinyTIFFWriter_Greyscale);
    if (!tif) {
        fmt::print("Error: Could not open TIFF\n");
        std::exit(1);
    }
    TinyTIFFWriter_writeImage(tif, data);
    TinyTIFFWriter_close(tif);
    fmt::print("Wrote {:2} bit {:5} image: {}\n", sizeof(T) * 8, type, filename);
}

template <typename T>
auto write_tiff(std::string filename_, int number, sycl::host_ptr<T> data) {
    return write_tiff(filename_, number, data.get());
}

int main(int argc, char** argv) {
    auto start_time = std::chrono::high_resolution_clock::now();

    // Parse arguments and get our H5Reader
    auto parser = FPGAArgumentParser();
    parser.add_h5read_arguments();
    auto args = parser.parse_args(argc, argv);
    auto reader = args.file.empty() ? H5Read() : H5Read(args.file);

    sycl::queue Q{args.device(), sycl::property::queue::enable_profiling{}};

    fmt::print("Running with {}{}-bit{} wide blocks\n", BOLD, BLOCK_SIZE * 16, NC);

    auto slow = reader.get_image_slow();
    auto fast = reader.get_image_fast();
    const size_t num_pixels = slow * fast;

    // Make sure these match our hardcoded values
    assert(slow == SLOW);
    assert(fast == FAST);

    // Mask data is the same for all images, so we copy it to device early
    auto mask_data = device_ptr<uint8_t>(malloc_device<uint8_t>(num_pixels, Q));
    // Declare the image data that will be remotely accessed
    auto image_data = host_ptr<uint16_t>(malloc_host<uint16_t>(num_pixels, Q));
    check_allocs(mask_data, image_data);
    // Paranoia: Ensure that this is properly aligned
    assert(reinterpret_cast<uintptr_t>(image_data.get()) % 64 == 0);

    fmt::print(
      "Block data:\n"
      "         SIZE: {} px per block\n"
      "    REMAINDER: {} px unprocessed per row\n"
      "  FULL_BLOCKS: {} blocks across image width\n",
      BLOCK_SIZE,
      BLOCK_REMAINDER,
      FULL_BLOCKS);

    if (reader.get_mask()) {
        // Calculate kernel-summed mask data
        auto mask_kernelsum =
          calculate_kernel_sum_sat(reader.get_mask()->data(), fast, slow);
        // draw_image_data(mask_kernelsum.get(), fast - 16, slow - 16, 16, 16, fast, slow);
        fmt::print("Uploading mask data to accelerator.... ");
        auto e_mask_upload = Q.submit(
          [&](handler& h) { h.memcpy(mask_data, mask_kernelsum.get(), num_pixels); });
        Q.wait();
        fmt::print("done in {:.1f} ms ({:.2f} GBps)\n",
                   event_ms(e_mask_upload),
                   event_GBps(e_mask_upload, num_pixels));
    } else {
        fmt::print("Setting empty mask...");
        auto e_mask_set = Q.memset(
          mask_data, 0, num_pixels * sizeof(decltype(mask_data)::element_type));
        e_mask_set.wait();
        fmt::print("done in {:.1f} ms ({:.2f} GBps)\n",
                   event_ms(e_mask_set),
                   event_GBps(e_mask_set, num_pixels));
    }

    FindSpotsDebugOutput debug_data{Q};

    // auto destination_data = host_ptr<uint16_t>(malloc_host<uint16_t>(num_pixels, Q));
    // auto destination_data_sq = host_ptr<uint16_t>(malloc_host<uint16_t>(num_pixels, Q));
    auto strong_pixels = host_ptr<bool>(malloc_host<bool>(num_pixels, Q));
    check_allocs(strong_pixels);
    memset(strong_pixels, 0, sizeof(bool) * num_pixels);

    // for (size_t i = 0; i < num_pixels; ++i) {
    //     strong_pixels[i] = false;
    // }

    Q.wait();

    fmt::print("Starting image loop:\n");
    for (int i = 0; i < reader.get_number_of_images(); ++i) {
        fmt::print("\nReading Image {}\n", i);
        reader.get_image_into(i, image_data);

        // Precalculate host-side the answers we expect, so we can validate
        fmt::print("Calculating host sum\n");
        // Now we are using blocks and discarding excess, do that here
        size_t host_sum = 0;
        for (int i = 0; i < FULL_BLOCKS * BLOCK_SIZE; ++i) {
            host_sum += image_data[i];
        }
        fmt::print("Starting Kernels\n");
        auto t1 = std::chrono::high_resolution_clock::now();

        event e_producer = run_producer(Q, image_data);
        event e_module = run_module(Q, mask_data, strong_pixels, debug_data);
        Q.wait();

        auto t2 = std::chrono::high_resolution_clock::now();
        double ms_all =
          std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count()
          * 1000;

        fmt::print(" ... produced in {:.2f} ms ({:.3g} GBps)\n",
                   event_ms(e_producer),
                   event_GBps(e_producer, num_pixels * sizeof(uint16_t) / 2));
        fmt::print(" ... consumed in {:.2f} ms ({:.3g} GBps)\n",
                   event_ms(e_module),
                   event_GBps(e_module, num_pixels * sizeof(uint16_t) / 2));
        fmt::print(" ... Total consumed + piped in host time {:.2f} ms ({:.3g} GBps)\n",
                   ms_all,
                   GBps(num_pixels * sizeof(uint16_t), ms_all));

        Q.wait();

        int x = 2101, y = 3228;
        // Print a section of the image and "destination" arrays
        fmt::print("Data:\n");
        draw_image_data(image_data, x, y, 16, 16, fast, slow);

        // fmt::print("\nMirror:\n");
        // draw_image_data(destination_data, 0, 0, 16, 16, fast, slow);

        // Count the strong pixels
        size_t num_strong = 0;
        for (size_t i = 0; i < num_pixels; ++i) {
            if (strong_pixels[i]) {
                num_strong++;
            }
        }
        fmt::print("Sum:\n");
        draw_image_data(debug_data.sum, x, y, 16, 16, fast, slow);

        fmt::print("SumSq:\n");
        draw_image_data(debug_data.sumsq, x, y, 16, 16, fast, slow);

        fmt::print("Mean:\n");
        draw_image_data(debug_data.mean, x, y, 16, 16, fast, slow);
        // fmt::print("Number of strong pixels: {}\n", num_strong);
        // fmt::print("\nSumSq:\n");
        // draw_image_data(strong_pixels, 0, 0, 16, 16, fast, slow);

        fmt::print("Variance:\n");
        draw_image_data(debug_data.variance, x, y, 16, 16, fast, slow);

        // fmt::print("Size: {}\n", sizeof(bool));
        // auto filename = fmt::format("image_{}.tif", i);
        // TinyTIFFWriterFile *tif = TinyTIFFWriter_open(filename.c_str(), 16, TinyTIFFWriter_UInt, 1, FAST, SLOW, TinyTIFFWriter_Greyscale);
        // if (!tif) {
        //     fmt::print("Error: Could not open TIFF\n");
        //     std::exit(1);
        // }
        // TinyTIFFWriter_writeImage(tif, image_data.get());
        // TinyTIFFWriter_close(tif);
        // for (int i = 0; i < num_pixels; ++i) {
        //     ((uint8_t*)strong_pixels.get())[i] *= 255;

        //     ((uint16_t*)image_data.get())[i] *= 20000;
        // }
        write_tiff("{}_image.tif", i, image_data.get());
        write_tiff("{}_strong.tif", i, strong_pixels.get());
#ifdef DEBUG_IMAGES
        write_tiff("{}_dispersion.tif", i, debug_data.dispersion);
        write_tiff("{}_mean.tif", i, debug_data.mean);
        write_tiff("{}_variance.tif", i, debug_data.variance);
#endif
        break;
    }

    free(image_data, Q);
    free(mask_data, Q);
    auto end_time = std::chrono::high_resolution_clock::now();

    fmt::print(
      "Total run duration: {:.2f} s\n",
      std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time)
        .count());
}
