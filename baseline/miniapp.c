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

double time_baselines(h5read_handle* obj,
                                    int n_images,
                                    void** spotfinders,
                                    int* full_results,
                                    int preload) {
    int n_images_to_use = (n_images > 20) ? 20 : n_images;
    if (preload) {
        int temp;
        image_t* images[n_images_to_use];
        for (int i=0; i<n_images_to_use; i++) images[i] = h5read_get_image(obj, i);
        double t0 = omp_get_wtime();
        for (size_t j = 0; j < n_images_to_use; j++) {
            temp = spotfinder_standard_dispersion(spotfinders[omp_get_thread_num()], images[j]);
            full_results[j] = temp;
        }
        for (int i=0; i<n_images_to_use; i++) h5read_free_image(images[i]);
        return ((double)n_images / (double)n_images_to_use) * (omp_get_wtime() - t0);
    } else {
        int temp;
        image_t* image;
        double t0 = omp_get_wtime();
        for (size_t j = 0; j < n_images_to_use; j++) {
            image = h5read_get_image(obj, j);
            temp = spotfinder_standard_dispersion(spotfinders[omp_get_thread_num()], image);
            h5read_free_image(image);
            full_results[j] = temp;
        }
        return ((double)n_images / (double)n_images_to_use) * (omp_get_wtime() - t0);

    }
}

double time_parallelism_over_images(h5read_handle *obj, int n_images, void** spotfinders, int* full_results, int preload) {
    if (preload) {
        int temp;
        image_t* images[n_images];
        for (int i=0; i<n_images; i++) images[i] = h5read_get_image(obj, i);
        double t0 = omp_get_wtime();
        #pragma omp parallel for default(none) private(temp) shared(n_images, obj, spotfinders, full_results, images) schedule(dynamic)
        for (size_t j=0; j<n_images; j++) {
            temp = spotfinder_standard_dispersion(spotfinders[omp_get_thread_num()], images[j]);
            full_results[j] = temp;
        }
        for (int i=0; i<n_images; i++) h5read_free_image(images[i]);
        return omp_get_wtime() - t0;
    } else {
        int temp;
        image_t* image;
        double t0 = omp_get_wtime();
        #pragma omp parallel for default(none) private(temp, image) shared(n_images, obj, spotfinders, full_results) schedule(dynamic)
        for (size_t j=0; j<n_images; j++) {
            image = h5read_get_image(obj, j);
            temp = spotfinder_standard_dispersion(spotfinders[omp_get_thread_num()], image);
            h5read_free_image(image);
            full_results[j] = temp;
        }
        return omp_get_wtime() - t0;
    }
}

double time_parallelism_over_modules(h5read_handle* obj, int n_images, int n_modules, void** mini_spotfinders, int* mini_results, int preload) {
    if (preload) {
        uint32_t strong_pixels_from_modules=0;
        image_modules_t* modules[n_images];
        for (int i=0; i<n_images; i++) modules[i] = h5read_get_image_modules(obj, i);
        double t0 = omp_get_wtime();
        for (size_t j=0; j<n_images; j++) {
            strong_pixels_from_modules = 0;
            image_modules_t *mods = modules[j];
    #pragma omp parallel for default(none) shared(n_images, mods, mini_spotfinders, n_modules, mini_results) reduction(+:strong_pixels_from_modules) schedule(dynamic)
            for (size_t n=0; n<n_modules; n++) {
                strong_pixels_from_modules += spotfinder_standard_dispersion_modules(mini_spotfinders[omp_get_thread_num()], mods, n);
            }
            mini_results[j] = strong_pixels_from_modules;
        }
        for (int i=0; i<n_images; i++) h5read_free_image_modules(modules[i]);
        return omp_get_wtime() - t0;
    } else {
        uint32_t strong_pixels_from_modules=0;
        image_modules_t* modules;
        double t0 = omp_get_wtime();
        for (size_t j=0; j<n_images; j++) {
            modules = h5read_get_image_modules(obj, j);
            strong_pixels_from_modules = 0;
    #pragma omp parallel for default(none) shared(n_images, modules, mini_spotfinders, n_modules, mini_results) reduction(+:strong_pixels_from_modules) schedule(dynamic)
            for (size_t n=0; n<n_modules; n++) {
                strong_pixels_from_modules += spotfinder_standard_dispersion_modules(mini_spotfinders[omp_get_thread_num()], modules, n);
            }
            h5read_free_image_modules(modules);
            mini_results[j] = strong_pixels_from_modules;
        }
        return omp_get_wtime() - t0;
    }
}

double time_parallelism_over_modules_noblit(h5read_handle* obj, int n_images, void** noblit_spotfinders, int* both_results_nb, int preload) {
    if (preload) {
        uint32_t result;
        image_t* images[n_images];
        for (int i=0; i<n_images; i++) images[i] = h5read_get_image(obj, i);
        double t0 = omp_get_wtime();
        for (size_t j=0; j<n_images; j++) {
            result = spotfinder_standard_dispersion_modules_new(noblit_spotfinders[omp_get_thread_num()], images[j]);
            both_results_nb[j] = result;
        }
        for (int i=0; i<n_images; i++) h5read_free_image(images[i]);
        return omp_get_wtime()-t0;
    } else {
        uint32_t result;
        image_t* image;
        double t0 = omp_get_wtime();
        for (size_t j=0; j<n_images; j++) {
            image = h5read_get_image(obj, j);
            result = spotfinder_standard_dispersion_modules_new(noblit_spotfinders[omp_get_thread_num()], image);
            both_results_nb[j] = result;
            h5read_free_image(image);
        }
        return omp_get_wtime()-t0;
    }
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

    int baseline_results[n_images];
    int full_results[n_images];
    int modules_results[n_images];
    int modules_results_nb[n_images];

    int preload_results[n_images];
    
    double baseline_time = time_baselines(obj, n_images, spotfinders, baseline_results, 1);
    double over_images_time = time_parallelism_over_images(obj, n_images, spotfinders, full_results, 1);
    double over_modules_time = time_parallelism_over_modules(obj, n_images, n_modules, mini_spotfinders, modules_results, 1);
    double over_modules_noblit_time = time_parallelism_over_modules_noblit(obj, n_images, noblit_spotfinders, modules_results_nb, 1);

    for (size_t j=0; j<num_spotfinders; j++) {
        spotfinder_free(mini_spotfinders[j]);
        spotfinder_free(spotfinders[j]);
        spotfinder_free_new(noblit_spotfinders[j]);
    }

    h5read_free(obj);

    printf(
        "\nTime to run with parallel over:\n\
        Nothing (baseline)             %4.0f ms/image  (%2.2fx speedup)\n\
        Images:                        %4.0f ms/image  (%2.2fx speedup)\n\
        Modules:                       %4.0f ms/image  (%2.2fx speedup)\n\
        Modules (no blit):             %4.0f ms/image  (%2.2fx speedup)\n",
        baseline_time / n_images * 1000,
        baseline_time / baseline_time,
        over_images_time / n_images * 1000,
        baseline_time / over_images_time,
        over_modules_time / n_images * 1000,
        baseline_time / over_modules_time,
        over_modules_noblit_time / n_images * 1000,
        baseline_time / over_modules_noblit_time
    );

    printf("\nStrong pixels count results:\n");
    printf("Img# Images Modules  No blit\n");
    int num_to_print = (n_images > 5) ? 5 : n_images;
    for (size_t j = 0; j < num_to_print; j++) {
        char *col = "\033[1;31m";
        if (omp_get_max_threads() % 2 == 0){
            if (full_results[j] == modules_results[j]) {
                col = "\033[32m";
            }
        } else if (full_results[j] == modules_results[j]) {
            col = "\033[32m";
        }
        printf("%s%4d %6d %7d \033[0m%s %6d\n\033[0m",
               col,
               j,
               full_results[j],
               modules_results[j],
               col,
               modules_results_nb[j]);
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
                    "Images "
                    "Modules "
                    "\"Modules (no blit)\"\n"
                );
            }
            fprintf(fp,
                "%d %d %4.0f %4.0f %4.0f\n",
                n_images,
                omp_get_max_threads(),
                over_images_time / n_images * 1000,
                over_modules_time / n_images * 1000,
                over_modules_noblit_time / n_images * 1000
            );
        }
        fclose(fp);
    }
    return 0;
}
