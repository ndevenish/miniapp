#include <assert.h>
#include <inttypes.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "baseline.h"
#include "h5read.h"
#include "eiger2xe.h"

int main(int argc, char **argv) {
    h5read_handle *obj = h5read_parse_standard_args(argc, argv);
    size_t n_images = h5read_get_number_of_images(obj);
    image_modules_t *modules = h5read_get_image_modules(obj, 0);

    size_t image_fast_size = E2XE_16M_FAST;
    size_t image_slow_size = E2XE_16M_SLOW;
    size_t module_fast_size = E2XE_MOD_FAST;
    size_t module_slow_size = E2XE_MOD_SLOW;
    size_t n_modules = E2XE_16M_NSLOW * E2XE_16M_NFAST; //modules->modules;
    printf("Num modules: %d\n", n_modules);

    int num_spotfinders;
    int thread_num;
#ifdef _OPENMP
    double tstart = omp_get_wtime();
    printf("OMP found; have %d threads\n", omp_get_max_threads());
    num_spotfinders = omp_get_max_threads();
#endif
#ifndef _OPENMP
    num_spotfinders = 1;
#endif
    void* mini_spotfinders[num_spotfinders];
    void* mini_spotfinders_f[num_spotfinders];
    void* spotfinders[num_spotfinders];
    for (size_t j=0; j<num_spotfinders; j++) {
        mini_spotfinders[j] = spotfinder_create(module_fast_size, module_slow_size);
        mini_spotfinders_f[j] = spotfinder_create_f(module_fast_size, module_slow_size);
        spotfinders[j] = spotfinder_create(image_fast_size, image_slow_size);
    }
    int test_count = 0;
    image_t* image;
    int temp;
    int intvar;
    if (argc > 2) {
        n_images = (sscanf(argv[2], "%d", &intvar) == 1) ? intvar : n_images;
    }

    printf("Finding spots in %d images\n", n_images);

    int full_results[n_images];
    int mini_results[n_images];
    int both_results[n_images];
    int mini_f_results[n_images];

    double load_time = 0;
    double compute_time = 0;
    double t0 = omp_get_wtime();

// Parallelism over images
#pragma omp parallel for default(none) private(image, temp) shared(n_images, obj, spotfinders, full_results) reduction(+:load_time, compute_time)
    for (size_t j=0; j<n_images; j++) {
        double lt0 = omp_get_wtime();
        image = h5read_get_image(obj, j);
        double lt1 = omp_get_wtime();
        temp = spotfinder_standard_dispersion(spotfinders[omp_get_thread_num()], image);
        load_time += (lt1-lt0);
        compute_time += omp_get_wtime()-lt1;
        h5read_free_image(image);
        full_results[j] = temp;
    }
    
    printf("Image: load:%g compute:%g\n", load_time/omp_get_max_threads(), compute_time/omp_get_max_threads());

    double t1 = omp_get_wtime();

// Parallelism over modules
    uint32_t strong_pixels_from_modules=0;
    size_t n;
    load_time = 0;
    compute_time = 0;
    for (size_t j=0; j<n_images; j++) {
        double lt0 = omp_get_wtime();
        modules = h5read_get_image_modules(obj, j);
        double lt1 = omp_get_wtime();
        strong_pixels_from_modules = 0;
#pragma omp parallel for default(none) shared(n_images, modules, mini_spotfinders, n_modules, mini_results) reduction(+:strong_pixels_from_modules)
        for (n=0; n<n_modules; n++) {
            strong_pixels_from_modules += spotfinder_standard_dispersion_modules(mini_spotfinders[omp_get_thread_num()], modules, n);
        }
        load_time += lt1-lt0;
        compute_time += omp_get_wtime() - lt1;
        h5read_free_image_modules(modules);
        mini_results[j] = strong_pixels_from_modules;
        //printf("New total for image %d: %d\n", j, strong_pixels_from_modules);
    }
    printf("Modules: load:%g compute:%g\n", load_time, compute_time);

    double t2 = omp_get_wtime();

// Parallelism over both
    if (omp_get_max_threads() % 2 == 0) {
        omp_set_nested(1);
        omp_set_max_active_levels(2);
    #pragma omp parallel for default(none) private(image, temp, modules, strong_pixels_from_modules, n) shared(n_images, n_modules, obj, mini_spotfinders, both_results) num_threads(2) //schedule(static,1)
        for (size_t j=0; j<n_images; j++) {
            modules = h5read_get_image_modules(obj, j);
            strong_pixels_from_modules = 0;
            int offset = (omp_get_max_threads()/2) * omp_get_thread_num();
    #pragma omp parallel for default(none) shared(modules, mini_spotfinders, n_modules,offset) reduction(+:strong_pixels_from_modules) num_threads(omp_get_max_threads()/2) //schedule(static, 8)
            for (n=0; n<n_modules; n++) {
                strong_pixels_from_modules += spotfinder_standard_dispersion_modules(mini_spotfinders[offset+omp_get_thread_num()], modules, n);
            }
            h5read_free_image_modules(modules);
            both_results[j] = strong_pixels_from_modules;
        }
    }

    double t3 = omp_get_wtime();

// Parallelism over modules with floats
    for (size_t j=0; j<n_images; j++) {
        modules = h5read_get_image_modules(obj, j);
        strong_pixels_from_modules = 0;
#pragma omp parallel for default(none) shared(n_images, modules, mini_spotfinders_f, n_modules, mini_f_results) reduction(+:strong_pixels_from_modules)
        for (n=0; n<n_modules; n++) {
            strong_pixels_from_modules += spotfinder_standard_dispersion_modules_f(mini_spotfinders_f[omp_get_thread_num()], modules, n);
        }
        h5read_free_image_modules(modules);
        mini_f_results[j] = strong_pixels_from_modules;
    }

    double t4 = omp_get_wtime();

// Parallelism over images, but using modules
#pragma omp parallel for default(none) private(image, temp, modules, strong_pixels_from_modules, n) shared(n_images, n_modules, obj, mini_spotfinders, both_results)
    for (size_t j=0; j<n_images; j++) {
        modules = h5read_get_image_modules(obj, j);
        strong_pixels_from_modules = 0;
        for (n=0; n<n_modules; n++) {
            strong_pixels_from_modules += spotfinder_standard_dispersion_modules(mini_spotfinders[omp_get_thread_num()], modules, n);
        }
        h5read_free_image_modules(modules);
        both_results[j] = strong_pixels_from_modules;
    }

    double t5 = omp_get_wtime();

    for (size_t j=0; j<omp_get_max_threads(); j++) {
        h5read_free_image(image_sample[j]);
    }

    printf(
      "\nTime to run with parallel over:\n\
Images with modules: %4.0f ms/image\n\
Images:              %4.0f ms/image\n\
Modules:             %4.0f ms/image\n\
Modules (float):     %4.0f ms/image\n\
Both:                %4.0f ms/image\n",
      (t5 - t4) / n_images * 1000,
      (t1 - t0) / n_images * 1000,
      (t2 - t1) / n_images * 1000,
      (t4 - t3) / n_images * 1000,
      (t3 - t2) / n_images * 1000);

    printf("\nStrong pixels count results:\n");
    printf("Img# Images Modules (float)  Both\n");
    for (size_t j = 0; j < 5; j++) {
        char *col = "\033[1;31m";
        if (full_results[j] == mini_results[j] && full_results[j] == both_results[j]) {
            col = "\033[32m";
        }
        printf("%s%4d %6d %7d \033[0m%6d%s %5d\n\033[0m",
               col,
               j,
               full_results[j],
               mini_results[j],
               mini_f_results[j],
               col,
               both_results[j]);
    }

    uint16_t image_slow = 0, image_fast = 0;
    void *spotfinder = NULL;
    size_t modules_true, image_true;
    for (size_t j = 0; j < 1; j++) {
        image = h5read_get_image(obj, j);
        modules = h5read_get_image_modules(obj, j);
        modules_true=0;
        image_true=0;

        if (j == 0) {
            // Need to wait until we have an image to get its size
            image_slow = image->slow;
            image_fast = image->fast;
            spotfinder = spotfinder_create(image_fast, image_slow);
            n = 0;
        } else {
            // For sanity sake, check this matches
            assert(image->slow == image_slow);
            assert(image->fast == image_fast);
        }

        for (n=0; n<modules->modules; n++) {
            size_t i_fast = n % 4;
            size_t i_slow = n / 4;
            size_t r_0 = i_slow * (modules->slow + 38) * image_fast;
            for (size_t k=0; k<modules->fast*modules->slow; k++) {
                size_t offset = r_0 + (k / modules->fast) * image->fast + i_fast *(modules->fast+12);
                size_t idx = offset + k%modules->fast;
                if (modules->data[n*(modules->fast*modules->slow)+k] != image->data[idx]) {
                    printf("%d %d %d\n", k/modules->fast, k%modules->fast, offset);
                    printf("Module %d: non-equal at %d-%d: %d %d\n", n, n*(modules->fast*modules->slow)+k, idx, modules->data[n*(modules->fast*modules->slow)+k], image->data[idx]);
                    break;//exit(1);
                }
            }
        }

        for (n=0; n<modules->modules; n++) {
            size_t i_fast = n % 4;
            size_t i_slow = n / 4;
            size_t r_0 = i_slow * (modules->slow + 38) * image_fast;
            modules_true = 0;
            image_true = 0;
            for (size_t k=0; k<modules->fast*modules->slow; k++) {
                size_t offset = r_0 + (k / modules->fast) * image->fast + i_fast *(modules->fast+12);
                size_t idx = offset + k%modules->fast;
                if (modules->mask[n*(modules->fast*modules->slow)+k] != image->mask[idx]) {
                    printf("Mask disagreement at %d %d\n", n*(modules->fast*modules->slow)+k, idx);
                }
            }
        }

        uint32_t strong_pixels;
        size_t zero;
        size_t zero_m;

        // for (int l = 0; l < 100; l++) {
        strong_pixels = spotfinder_standard_dispersion(spotfinder, image);

        zero = 0;
        for (size_t i = 0; i < (image->fast * image->slow); i++) {
            if (image->data[i] == 0 && image->mask[i] == 1) {
                zero++;
            }
        }

        zero_m = 0;
        for (size_t i = 0; i < (modules->fast * modules->slow * modules->modules);
             i++) {
            if (modules->data[i] == 0 && modules->mask[i] == 1) {
                zero_m++;
            }
        }
        // }

        printf("image %ld had %ld / %ld valid zero pixels, %" PRIu32 " strong pixels\n\n",
               j,
               zero,
               zero_m,
               strong_pixels);

        // h5read_free_image_modules(modules);
        h5read_free_image(image);
    }

    for (size_t j=0; j<num_spotfinders; j++) {
        spotfinder_free(mini_spotfinders[j]);
        spotfinder_free(mini_spotfinders_f[j]);
        spotfinder_free(spotfinders[j]);
    }

    h5read_free_image_modules(modules);
    //spotfinder_free(mini_spotfinder);
    spotfinder_free(spotfinder);
    h5read_free(obj);

    return 0;
}
