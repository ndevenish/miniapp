#include "h5read.h"

#include <assert.h>
// #include <hdf5.h>
#include <libgen.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "eiger2xe.h"

// VDS stuff

#define MAXFILENAME 256
#define MAXDATAFILES 100
#define MAXDIM 3

typedef struct h5_data_file {
    char filename[MAXFILENAME];
    char dsetname[MAXFILENAME];
    int64_t file;
    int64_t dataset;
    size_t frames;
    size_t offset;
} h5_data_file;

struct _h5read_handle {
    int64_t master_file;
    int data_file_count;
    h5_data_file *data_files;
    size_t frames;  ///< Number of frames in this dataset
    size_t slow;    ///< Pixel dimension of images in the slow direction
    size_t fast;    ///< Pixel dimensions of images in the fast direction

    uint8_t *mask;         ///< Shared image mask
    uint8_t *module_mask;  ///< Shared module mask
    size_t mask_size;      ///< Total size(in pixels) of mask
};

void h5read_free(h5read_handle *obj) {
    // for (int i = 0; i < obj->data_file_count; i++) {
    //     H5Dclose(obj->data_files[i].dataset);
    //     H5Fclose(obj->data_files[i].file);
    // }
    if (obj->data_files) free(obj->data_files);
    // if (obj->master_file) H5Fclose(obj->master_file);

    free(obj->mask);
    free(obj->module_mask);

    free(obj);
}

/// Get the number of frames available
size_t h5read_get_number_of_images(h5read_handle *obj) {
    return obj->frames;
}

size_t h5read_get_image_slow(h5read_handle *obj) {
    return obj->slow;
}

size_t h5read_get_image_fast(h5read_handle *obj) {
    return obj->fast;
}

void h5read_free_image(image_t *i) {
    free(i->data);
    // Mask is a pointer to the file-global file mask so isn't freed
    free(i);
}

uint8_t *h5read_get_mask(h5read_handle *obj) {
    return obj->mask;
}

/// blit the relevent pixel data across from a single image into a collection
/// of image modules - will allocate the latter
///
/// @param image    The image to blit from
/// @param modules  The modules object to fill
void _blit(image_t *image, image_modules_t *modules) {
    // Number of modules in fast, slow directions
    size_t fast, slow;
    if (image->slow == E2XE_16M_SLOW) {
        fast = 4;
        slow = 8;
    } else {
        fast = 2;
        slow = 4;
    }

    modules->slow = E2XE_MOD_SLOW;
    modules->fast = E2XE_MOD_FAST;
    modules->modules = slow * fast;

    size_t module_pixels = E2XE_MOD_SLOW * E2XE_MOD_FAST;

    modules->data = (uint16_t *)malloc(sizeof(uint16_t) * slow * fast * module_pixels);

    for (size_t _slow = 0; _slow < slow; _slow++) {
        size_t row0 = _slow * (E2XE_MOD_SLOW + E2XE_GAP_SLOW) * image->fast;
        for (size_t _fast = 0; _fast < fast; _fast++) {
            for (size_t row = 0; row < E2XE_MOD_SLOW; row++) {
                size_t offset =
                  (row0 + row * image->fast + _fast * (E2XE_MOD_FAST + E2XE_GAP_FAST));
                size_t target =
                  (_slow * fast + _fast) * module_pixels + row * E2XE_MOD_FAST;
                memcpy((void *)&modules->data[target],
                       (void *)&image->data[offset],
                       sizeof(uint16_t) * E2XE_MOD_FAST);
            }
        }
    }
}

image_modules_t *h5read_get_image_modules(h5read_handle *obj, size_t n) {
    image_t *image = h5read_get_image(obj, n);
    image_modules_t *modules = malloc(sizeof(image_modules_t));
    modules->data = NULL;
    modules->mask = obj->module_mask;
    modules->modules = -1;
    modules->fast = -1;
    modules->slow = -1;
    _blit(image, modules);
    h5read_free_image(image);
    return modules;
}

void h5read_free_image_modules(image_modules_t *i) {
    free(i->data);
    // Like image, mask is held on the central h5read_handle object
    free(i);
}

#define NUM_SAMPLE_IMAGES 4

/// Generate a sample image from number
void _generate_sample_image(h5read_handle *obj, size_t n, image_t_type *data) {
    assert(n >= 0 && n <= NUM_SAMPLE_IMAGES);

    if (n == 0) {
        memset(data, 0, E2XE_16M_FAST * E2XE_16M_SLOW * sizeof(uint16_t));
    } else if (n == 1) {
        // Image 1: I=1 for every unmasked pixel
        memset(data, 0, E2XE_16M_FAST * E2XE_16M_SLOW * sizeof(uint16_t));
        for (int mody = 0; mody < E2XE_16M_NSLOW; ++mody) {
            // row0 is the row of the module top row
            size_t row0 = mody * (E2XE_MOD_SLOW + E2XE_GAP_SLOW);
            for (int modx = 0; modx < E2XE_16M_NFAST; ++modx) {
                // col0 is the column of the module left
                int col0 = modx * (E2XE_MOD_FAST + E2XE_GAP_FAST);
                for (int row = 0; row < E2XE_MOD_SLOW; ++row) {
                    for (int x = 0; x < E2XE_MOD_FAST; ++x) {
                        *(data + E2XE_16M_FAST * (row0 + row) + col0 + x) = 1;
                    }
                }
            }
        }
    } else if (n == 2) {
        // Image 2: High pixel (100) every 42 pixels across the detector
        memset(data, 0, E2XE_16M_FAST * E2XE_16M_SLOW * sizeof(uint16_t));
        for (int y = 0; y < E2XE_16M_SLOW; y += 42) {
            for (int x = 0; x < E2XE_16M_FAST; x += 42) {
                int k = y * E2XE_16M_FAST + x;
                data[k] = 100;
            }
        }
    } else if (n == 3) {
        // Image 3: "Random" background, zero on masks

        // Implement a very simple 'random' generator, Numerical Methods' ranqd1
        // - this ensures that we have stable cross-platform results.
        uint64_t idum = 0;

        memset(data, 0, E2XE_16M_FAST * E2XE_16M_SLOW * sizeof(uint16_t));
        for (int mody = 0; mody < E2XE_16M_NSLOW; ++mody) {
            // row0 is the row of the module top row
            size_t row0 = mody * (E2XE_MOD_SLOW + E2XE_GAP_SLOW);
            for (int modx = 0; modx < E2XE_16M_NFAST; ++modx) {
                // col0 is the column of the module left
                int col0 = modx * (E2XE_MOD_FAST + E2XE_GAP_FAST);
                for (int row = 0; row < E2XE_MOD_SLOW; ++row) {
                    for (int x = 0; x < E2XE_MOD_FAST; ++x) {
                        *(data + E2XE_16M_FAST * (row0 + row) + col0 + x) = (idum % 10);
                        // basic LCG % 4 isn't unpredictable enough for us. Fake it.
                        do {
                            idum = 1664525UL * idum + 1013904223UL;
                        } while (idum % 10 >= 4);
                    }
                }
            }
        }
    } else {
        fprintf(stderr, "Error: Unhandled sample image %d\n", (int)n);
        exit(2);
    }
}

void h5read_get_image_into(h5read_handle *obj, size_t index, image_t_type *data) {
    if (index >= obj->frames) {
        fprintf(stderr,
                "Error: image %ld greater than number of frames (%ld)\n",
                index,
                obj->frames);
        exit(1);
    }

    // Check if we are using sample data
    if (obj->data_files == 0) {
        // We are using autogenerated image data. Return that.
        _generate_sample_image(obj, index, data);
        return;
    }

    fprintf(stderr,
            "Error: H5 capabilities have been removed. Only sample data supported.");
    exit(1);
}

image_t *h5read_get_image(h5read_handle *obj, size_t n) {
    // Make an image_t to write into
    image_t *result = malloc(sizeof(image_t));
    result->mask = obj->mask;
    result->fast = obj->fast;
    result->slow = obj->slow;
    // Create the buffer here. This will be freed by h5read_free_image
    result->data = malloc(sizeof(image_t_type) * obj->slow * obj->fast);
    // Use our read-into-buffer function to fill this
    h5read_get_image_into(obj, n, result->data);

    return result;
}

// void setup_data(h5read_handle *obj) {
//     int64_t dataset = obj->data_files[0].dataset;
//     int64_t datatype = H5Dget_type(dataset);

//     if (H5Tget_size(datatype) != 2) {
//         fprintf(stderr, "native data size != 2 (%ld)\n", H5Tget_size(datatype));
//         exit(1);
//     }

//     int64_t space = H5Dget_space(dataset);

//     if (H5Sget_simple_extent_ndims(space) != 3) {
//         fprintf(stderr, "raw data not three dimensional\n");
//         exit(1);
//     }

//     hsize_t dims[3];
//     H5Sget_simple_extent_dims(space, dims, NULL);

//     obj->slow = dims[1];
//     obj->fast = dims[2];

//     printf("Total data size: %ldx%ldx%ld\n", obj->frames, obj->slow, obj->fast);
//     H5Sclose(space);
// }

// Generate a mask with just module bounds masked off
uint8_t *_generate_e2xe_16m_mask() {
    assert(E2XE_MOD_SLOW * E2XE_16M_NSLOW + E2XE_GAP_SLOW * (E2XE_16M_NSLOW - 1)
           == E2XE_16M_SLOW);
    assert(E2XE_MOD_FAST * E2XE_16M_NFAST + E2XE_GAP_FAST * (E2XE_16M_NFAST - 1)
           == E2XE_16M_FAST);
    uint8_t *mask = calloc(E2XE_16M_SLOW * E2XE_16M_SLOW, sizeof(uint8_t));
    for (size_t i = 0; i < E2XE_16M_SLOW * E2XE_16M_SLOW; ++i) {
        mask[i] = 1;
    }
    // Horizontal gaps
    for (int gap = 1; gap < E2XE_16M_NSLOW; ++gap) {
        // First gap has 1 module 0 gap, second gap has 2 modules 1 gap etc
        size_t y = gap * E2XE_MOD_SLOW + (gap - 1) * E2XE_GAP_SLOW;
        // Horizontal gaps can just be bulk memset for each gap
        memset(mask + y * E2XE_16M_FAST, 0, E2XE_GAP_SLOW * E2XE_16M_FAST);
    }
    // Vertical gaps
    for (int gap = 1; gap < E2XE_16M_NFAST; ++gap) {
        // First gap has 1 module 0 gap, second gap has 2 modules 1 gap etc
        size_t x = gap * E2XE_MOD_FAST + (gap - 1) * E2XE_GAP_FAST;
        for (int y = 0; y < E2XE_16M_SLOW; ++y) {
            memset(mask + y * E2XE_16M_FAST + x, 0, E2XE_GAP_FAST);
        }
    }
    return mask;
}

h5read_handle *h5read_generate_samples() {
    h5read_handle *file = calloc(1, sizeof(h5read_handle));

    // Generate the mask - with module gaps masked off
    file->slow = E2XE_16M_SLOW;
    file->fast = E2XE_16M_FAST;
    file->mask = _generate_e2xe_16m_mask();
    // Module mask is just empty for now
    file->module_mask =
      malloc(E2XE_16M_NSLOW * E2XE_16M_NFAST * E2XE_MOD_FAST * E2XE_MOD_SLOW);
    for (size_t i = 0;
         i < E2XE_16M_NSLOW * E2XE_16M_NFAST * E2XE_MOD_FAST * E2XE_MOD_SLOW;
         ++i) {
        file->module_mask[i] = 1;
    }

    // Debug - writing mask to a test file
    // FILE *fo = fopen("mask.dat", "w");
    // fwrite(file->mask, sizeof(uint8_t), E2XE_16M_SLOW * E2XE_16M_FAST, fo);
    // fclose(fo);

    file->frames = NUM_SAMPLE_IMAGES;
    return file;
}

h5read_handle *h5read_parse_standard_args(int argc, char **argv) {
    bool implicit_sample = getenv("H5READ_IMPLICIT_SAMPLE") != NULL;
    const char *USAGE = implicit_sample
                          ? "Usage: %s [-h|--help] [-v] [FILE.nxs | --sample]"
                          : "Usage: %s [-h|--help] [-v] (FILE.nxs | --sample)";
    const char *HELP =
      "Options:\n\
  FILE.nxs      Path to the Nexus file to parse\n\
  -h, --help    Show this message\n\
  -v            Verbose HDF5 message output\n\
  --sample      Don't load a data file, instead use generated test data.\n\
                If H5READ_IMPLICIT_SAMPLE is set, then this is assumed,\n\
                if a file is not provided.";

    bool verbose = false;
    bool sample_data = false;

    // Handle simple case of -h or --help
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            fprintf(stderr, USAGE, argv[0]);
            fprintf(stderr, "\n\n%s\n", HELP);
            exit(0);
        }
        if (!strcmp(argv[i], "-v")) {
            verbose = true;
            // Shift the rest over this one so that we only have positionals
            for (int j = i; j < argc; j++) {
                argv[i] = argv[j];
            }
            argc -= 1;
        }
        if (!strcmp(argv[i], "--sample")) {
            sample_data = true;
            // Shift the rest over this one so that we only have positionals
            for (int j = i; j < argc; j++) {
                argv[i] = argv[j];
            }
            argc -= 1;
        }
    }
    // if (!verbose) {
    //     // Turn off verbose hdf5 errors
    //     H5Eset_auto(H5E_DEFAULT, NULL, NULL);
    // }
    bool implicit_sample_data = false;
    if (implicit_sample && argc == 1 && !sample_data) {
        fprintf(
          stderr,
          "No input file but H5READ_IMPLICIT_SAMPLE is set - defaulting to sample "
          "data\n");
        sample_data = true;
    }

    if (argc == 1 && !sample_data) {
        fprintf(stderr, USAGE, argv[0]);
        exit(1);
    }
    // If we specifically requested --sample, then we can have no more arguments
    if (argc > 1 && sample_data) {
        fprintf(stderr, "Unrecognised extra arguments with --sample\n");
        exit(1);
    }

    h5read_handle *handle = 0;
    if (sample_data) {
        fprintf(stderr, "Using SAMPLE dataset\n");
        handle = h5read_generate_samples();
    } else {
        // handle = h5read_open(argv[1]);
        fprintf(
          stderr,
          "Error: H5 capabilities have been removed. Only sample data supported.");
        exit(1);
    }
    if (handle == NULL) {
        fprintf(stderr, "Error: Could not open nexus file %s\n", argv[1]);
        exit(1);
    }
    return handle;
}