#include <assert.h>
#include <inttypes.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "baseline.h"
#include "h5read.h"
#include "eiger2xe.h"

void time_image_loading(h5read_handle* obj, int n_images) {
    image_modules_t* modules;
    image_t* image;
    double t0 = omp_get_wtime();
    for (size_t j=0; j<n_images; j++) {
        image = h5read_get_image(obj, j);
        h5read_free_image(image);
    }
    double t1 = omp_get_wtime();
    for (size_t j=0; j<n_images; j++) {
        modules = h5read_get_image_modules(obj, j);
        h5read_free_image_modules(modules);
    }
    double t2 = omp_get_wtime();
    printf("For %d images:\n"
            "Image load time:  %4.0fms/image\n"
            "Module load time: %4.0fms/image\n",
            n_images,
            1000*(t1-t0)/n_images, 
            1000*(t2-t1)/n_images);
}

double time_parallelism_over_images(h5read_handle *obj, int n_images, void** spotfinders, int* full_results) {
    int temp;
    image_t* image;
    double t0 = omp_get_wtime();
    #pragma omp parallel for default(none) private(image, temp) shared(n_images, obj, spotfinders, full_results)
    for (size_t j=0; j<n_images; j++) {
        image = h5read_get_image(obj, j);
        temp = spotfinder_standard_dispersion(spotfinders[omp_get_thread_num()], image);
        h5read_free_image(image);
        full_results[j] = temp;
    }
    return omp_get_wtime() - t0;
}

double time_parallelism_over_images_using_modules(h5read_handle *obj, int n_images, int n_modules, void** mini_spotfinders, int* full_results_m) {
    uint32_t strong_pixels_from_modules=0;
    image_modules_t* modules;
    double t0 = omp_get_wtime();
    #pragma omp parallel for default(none) private(modules, strong_pixels_from_modules) shared(n_images, n_modules, obj, mini_spotfinders, full_results_m)
    for (size_t j=0; j<n_images; j++) {
        modules = h5read_get_image_modules(obj, j);
        strong_pixels_from_modules = 0;
        for (size_t n=0; n<n_modules; n++) {
            strong_pixels_from_modules += spotfinder_standard_dispersion_modules(mini_spotfinders[omp_get_thread_num()], modules, n);
        }
        h5read_free_image_modules(modules);
        full_results_m[j] = strong_pixels_from_modules;
    }
    return omp_get_wtime() - t0;
}

double time_parallelism_over_modules(h5read_handle* obj, int n_images, int n_modules, void** mini_spotfinders, int* mini_results) {
    uint32_t strong_pixels_from_modules=0;
    image_modules_t* modules;
    double t0 = omp_get_wtime();
    for (size_t j=0; j<n_images; j++) {
        modules = h5read_get_image_modules(obj, j);
        strong_pixels_from_modules = 0;
#pragma omp parallel for default(none) shared(n_images, modules, mini_spotfinders, n_modules, mini_results) reduction(+:strong_pixels_from_modules)
        for (size_t n=0; n<n_modules; n++) {
            strong_pixels_from_modules += spotfinder_standard_dispersion_modules(mini_spotfinders[omp_get_thread_num()], modules, n);
        }
        h5read_free_image_modules(modules);
        mini_results[j] = strong_pixels_from_modules;
    }
    return omp_get_wtime() - t0;
}

double time_parallelism_over_both(h5read_handle* obj, int n_images, int n_modules, void** mini_spotfinders, int* both_results, int n_outer) {
    uint32_t strong_pixels_from_modules=0;
    image_modules_t* modules;
    double t0 = omp_get_wtime();
    int outer_num = (n_outer > omp_get_max_threads()) ? omp_get_max_threads() : n_outer;
    if (omp_get_max_threads() % outer_num == 0 && omp_get_max_threads()>1) {
        omp_set_nested(1); 
        omp_set_max_active_levels(2);
    }
    #pragma omp parallel for default(none) private(modules, strong_pixels_from_modules) shared(n_images, n_modules, obj, mini_spotfinders, both_results, outer_num) num_threads(outer_num)
        for (size_t j=0; j<n_images; j++) {
            modules = h5read_get_image_modules(obj, j);
            strong_pixels_from_modules = 0;
            int offset = (omp_get_max_threads()/outer_num) * omp_get_thread_num();
    #pragma omp parallel for default(none) shared(modules, mini_spotfinders, n_modules,offset) reduction(+:strong_pixels_from_modules) num_threads(omp_get_max_threads()/outer_num)
            for (size_t n=0; n<n_modules; n++) {
                strong_pixels_from_modules += spotfinder_standard_dispersion_modules(mini_spotfinders[offset+omp_get_thread_num()], modules, n);
            }
            h5read_free_image_modules(modules);
            both_results[j] = strong_pixels_from_modules;
        }
    return omp_get_wtime() - t0;
}

double time_parallelism_over_images_using_modules_noblit(h5read_handle* obj, int n_images, void** noblit_spotfinders, int* full_results_m_nb) {
    uint32_t result;
    image_t* image;
    double t0 = omp_get_wtime();
    #pragma omp parallel for default(none) private(result, image) shared(obj, noblit_spotfinders, full_results_m_nb, n_images)
    for (size_t j=0; j<n_images; j++) {
        image = h5read_get_image(obj, j);
        result = spotfinder_standard_dispersion_modules_new(noblit_spotfinders[omp_get_thread_num()], image);
        full_results_m_nb[j] = result;
        h5read_free_image(image);
    }
    return omp_get_wtime()-t0;
}

double time_parallelism_over_both_noblit(h5read_handle* obj, int n_images, void** noblit_spotfinders, int* both_results_nb, int n_outer) {
    uint32_t result;
    image_t* image;
    double t0 = omp_get_wtime();
    int outer_num = (n_outer > omp_get_max_threads()) ? omp_get_max_threads() : n_outer;
    outer_num = (outer_num > 2) ? 2 : outer_num;
    if (omp_get_max_threads() % outer_num == 0 && omp_get_max_threads()>1) {
        omp_set_nested(1); 
        omp_set_max_active_levels(2);
    }
    #pragma omp parallel for default(none) private(result, image) shared(obj, noblit_spotfinders, both_results_nb, n_images, outer_num) num_threads(outer_num)
    for (size_t j=0; j<n_images; j++) {
        int offset = (omp_get_max_threads()/outer_num) * omp_get_thread_num();
        image = h5read_get_image(obj, j);
        result = spotfinder_standard_dispersion_modules_new(noblit_spotfinders[offset+omp_get_thread_num()], image);
        both_results_nb[j] = result;
        h5read_free_image(image);
    }
    return omp_get_wtime()-t0;
}

int module_to_image_index(int module_num, int module_idx) {
    size_t i_fast = module_num % E2XE_16M_NFAST;
    size_t i_slow = module_num / E2XE_16M_NFAST;
    size_t r_0 = i_slow * (E2XE_MOD_SLOW + E2XE_GAP_SLOW) * E2XE_16M_FAST;
    size_t offset = r_0 + (module_idx / E2XE_MOD_FAST) * E2XE_16M_FAST + i_fast *(E2XE_MOD_FAST + E2XE_GAP_FAST);
    size_t image_idx = offset + module_idx % E2XE_MOD_FAST;
    return image_idx;
}

int main(int argc, char **argv) {

    h5read_handle *obj = h5read_parse_standard_args(argc, argv);
    size_t n_images = h5read_get_number_of_images(obj);
    image_modules_t *modules = h5read_get_image_modules(obj, 0);

    size_t image_fast_size = E2XE_16M_FAST;
    size_t image_slow_size = E2XE_16M_SLOW;
    size_t module_fast_size = E2XE_MOD_FAST;
    size_t module_slow_size = E2XE_MOD_SLOW;
    size_t n_modules = E2XE_16M_NSLOW * E2XE_16M_NFAST;
    printf("Num modules: %d\n", n_modules);

    int num_spotfinders;
#ifdef _OPENMP
    printf("OMP found; have %d threads\n", omp_get_max_threads());
    num_spotfinders = omp_get_max_threads();
#endif
#ifndef _OPENMP
    num_spotfinders = 1;
#endif
    void* mini_spotfinders[num_spotfinders];
    void* spotfinders[num_spotfinders];
    void* noblit_spotfinders[num_spotfinders];
    for (size_t j=0; j<num_spotfinders; j++) {
        mini_spotfinders[j] = spotfinder_create(module_fast_size, module_slow_size);
        spotfinders[j] = spotfinder_create(image_fast_size, image_slow_size);
        noblit_spotfinders[j] = spotfinder_create_new(image_fast_size, image_slow_size);
    }
    int test_count = 0;
    image_t* image;
    int temp;
    if (argc > 2) {
        n_images = (sscanf(argv[2], "%d", &temp) == 1) ? temp : n_images;
    }

    char* output_name;
    int write_output = 0;
    if (argc > 3) {
        write_output = 1;
        output_name = argv[3];
    }
    if (write_output) printf("Output file: %s\n", output_name);

    printf("Finding spots in %d images\n", n_images);

    // time_image_loading(obj, n_images);

    int full_results[n_images];
    int mini_results[n_images];
    int both_results[n_images];
    int full_results_m[n_images];
    int both_results_nb[n_images];

    double over_images_time = time_parallelism_over_images(obj, n_images, spotfinders, full_results);
    double over_images_using_modules_time = time_parallelism_over_images_using_modules(obj, n_images, n_modules, mini_spotfinders, full_results_m);
    double over_modules_time = time_parallelism_over_modules(obj, n_images, n_modules, mini_spotfinders, mini_results);
    double over_both_time = time_parallelism_over_both(obj, n_images, n_modules, mini_spotfinders, both_results, 2);
    double over_both_noblit_time = time_parallelism_over_both_noblit(obj, n_images, noblit_spotfinders, both_results_nb, 2); // Hard to change outer num properly

    for (size_t j=0; j<num_spotfinders; j++) {
        spotfinder_free(mini_spotfinders[j]);
        spotfinder_free(spotfinders[j]);
        spotfinder_free_new(noblit_spotfinders[j]);
    }

    h5read_free(obj);

    printf(
        "\nTime to run with parallel over:\n\
        Images with modules:           %4.0f ms/image\n\
        Images:                        %4.0f ms/image\n\
        Modules:                       %4.0f ms/image\n\
        Both:                          %4.0f ms/image\n\
        Both (no blit):                %4.0f ms/image\n",
        over_images_using_modules_time / n_images * 1000,
        over_images_time / n_images * 1000,
        over_modules_time / n_images * 1000,
        over_both_time / n_images * 1000,
        over_both_noblit_time / n_images * 1000
    );

    printf("\nStrong pixels count results:\n");
    printf("Img# Images Modules  Both  No blit\n");
    for (size_t j = 0; j < 5; j++) {
        char *col = "\033[1;31m";
        if (omp_get_max_threads() % 2 == 0){
            if (full_results[j] == mini_results[j] && full_results[j] == both_results[j] && full_results[j] == full_results_m[j]) {
                col = "\033[32m";
            }
        } else if (full_results[j] == mini_results[j] && full_results[j] == full_results_m[j]) {
            col = "\033[32m";
        }
        printf("%s%4d %6d %7d \033[0m%s %5d \033[33m%6d\n\033[0m",
               col,
               j,
               full_results[j],
               mini_results[j],
               col,
               both_results[j],
               both_results_nb[j]);
    }

    if (write_output) {
        FILE* fp = fopen(output_name, "a");
        if (fp != NULL) {
            fseek(fp, 0, SEEK_END);
            long size = ftell(fp);
            if (size == 0) {
                fprintf(fp, 
                    "#Images "
                    "P "
                    "\"Images with modules\" "
                    "Images "
                    "Modules "
                    "Both "
                    "\"Both (no blit)\"\n"
                );
            }
            fprintf(fp,
                "%d %d %4.0f %4.0f %4.0f %4.0f %4.0f %4.0f\n",
                n_images,
                omp_get_max_threads(),
                over_images_using_modules_time / n_images * 1000,
                over_images_time / n_images * 1000,
                over_modules_time / n_images * 1000,
                over_both_time / n_images * 1000,
                over_both_noblit_time / n_images * 1000
            );
        }
        fclose(fp);
    }

    return 0;
}
