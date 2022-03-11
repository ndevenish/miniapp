#ifndef BASELINE_H
#define BASELINE_H

#include "h5read.h"

#ifdef __cplusplus
extern "C" {
#endif
void* spotfinder_create(size_t width, size_t height);
void spotfinder_free(void* context);
void* spotfinder_create_f(size_t width, size_t height);
void spotfinder_free_f(void* context);
void* spotfinder_create_new(size_t width, size_t height);
void spotfinder_free_new(void* context);
uint32_t spotfinder_standard_dispersion(void* context, image_t* image);
uint32_t spotfinder_standard_dispersion_modules(void* context,
                                                image_modules_t* image_modules,
                                                size_t index);  //, image_t* image);
uint32_t spotfinder_standard_dispersion_modules_f(void* context,
                                                  image_modules_t* image_modules,
                                                  size_t index);
uint32_t spotfinder_standard_dispersion_modules_new(void* context, image_t* image);
#ifdef __cplusplus
}
#endif

#endif