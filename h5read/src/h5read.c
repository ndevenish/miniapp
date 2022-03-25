#include "h5read.h"

#include <assert.h>
#include <hdf5.h>
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
    hid_t file;
    hid_t dataset;
    size_t frames;
    size_t offset;
} h5_data_file;

struct _h5read_handle {
    hid_t master_file;
    int data_file_count;
    h5_data_file *data_files;
    size_t frames;  ///< Number of frames in this dataset
    size_t slow;    ///< Pixel dimension of images in the slow direction
    size_t fast;    ///< Pixel dimensions of images in the fast direction

    uint8_t *mask;         ///< Shared image mask
    uint8_t *module_mask;  ///< Shared module mask
    size_t mask_size;      ///< Total size(in pixels) of mask
};

struct _h5read_handle_precalc {
    h5read_handle *obj;
    int *mask_kernels;
};

void h5read_free(h5read_handle *obj) {
    for (int i = 0; i < obj->data_file_count; i++) {
        H5Dclose(obj->data_files[i].dataset);
        H5Fclose(obj->data_files[i].file);
    }
    if (obj->data_files) free(obj->data_files);
    if (obj->master_file) H5Fclose(obj->master_file);

    free(obj->mask);
    free(obj->module_mask);

    free(obj);
}

void h5read_free_precalc(h5read_handle_precalc *obj_precalc, int free_base_obj) {
    if (free_base_obj) h5read_free(obj_precalc->obj);
    free(obj_precalc->mask_kernels);
    free(obj_precalc);
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

void h5read_free_image_and_mask_kernels(image_precalc_mask_t *i) {
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
        fprintf(stderr, "Error: Unhandled sample image %d\n", n);
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

    /* first find the right data file - having to do this lookup is annoying
       but probably cheap */
    int data_file;
    for (data_file = 0; data_file < obj->data_file_count; data_file++) {
        if ((index - obj->data_files[data_file].offset)
            < obj->data_files[data_file].frames) {
            break;
        }
    }

    if (data_file == obj->data_file_count) {
        fprintf(stderr, "Error: Could not find data file for frame %ld\n", index);
        exit(1);
    }

    h5_data_file *current = &(obj->data_files[data_file]);

    hid_t space = H5Dget_space(current->dataset);
    hid_t datatype = H5Dget_type(current->dataset);

    hsize_t block[3] = {1, obj->slow, obj->fast};
    hsize_t offset[3] = {index - current->offset, 0, 0};

    // select data to read #todo add status checks
    H5Sselect_hyperslab(space, H5S_SELECT_SET, offset, NULL, block, NULL);
    hid_t mem_space = H5Screate_simple(3, block, NULL);

    if (H5Dread(current->dataset, datatype, mem_space, space, H5P_DEFAULT, data) < 0) {
        H5Eprint(H5E_DEFAULT, NULL);
        exit(1);
    }

    H5Sclose(space);
    H5Sclose(mem_space);
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

image_precalc_mask_t *h5read_get_image_and_mask_kernels(h5read_handle_precalc *obj_precalc, size_t n) {
    // Make an image_t to write into
    image_precalc_mask_t *result = malloc(sizeof(image_precalc_mask_t));
    // h5read_handle *obj = obj_precalc->obj;
    result->mask = obj_precalc->obj->mask;
    result->fast = obj_precalc->obj->fast;
    result->slow = obj_precalc->obj->slow;
    // Create the buffer here. This will be freed by h5read_free_image
    result->data = malloc(sizeof(image_t_type) * obj_precalc->obj->slow * obj_precalc->obj->fast);
    // Use our read-into-buffer function to fill this
    h5read_get_image_into(obj_precalc->obj, n, result->data);

    result->mask_kernels = obj_precalc->mask_kernels;

    return result;
}

void read_mask(h5read_handle *obj) {
    char mask_path[] = "/entry/instrument/detector/pixel_mask";

    hid_t mask_dataset = H5Dopen(obj->master_file, mask_path, H5P_DEFAULT);

    if (mask_dataset < 0) {
        fprintf(stderr, "Error: While reading mask from %s\n", mask_path);
        exit(1);
    }

    hid_t datatype = H5Dget_type(mask_dataset);
    hid_t mask_info = H5Dget_space(mask_dataset);

    size_t mask_dsize = H5Tget_size(datatype);
    if (mask_dsize == 4) {
        printf("mask dtype uint32\n");
    } else if (mask_dsize == 8) {
        printf("mask dtype uint64\n");
    } else {
        fprintf(stderr, "Error: mask data size (%ld) != 4,8\n", H5Tget_size(datatype));
        exit(1);
    }

    obj->mask_size = H5Sget_simple_extent_npoints(mask_info);

    printf("Mask has %ld elements\n", obj->mask_size);

    void *buffer = NULL;

    uint32_t *raw_mask = NULL;
    uint64_t *raw_mask_64 = NULL;  // why?
    if (mask_dsize == 4) {
        raw_mask = (uint32_t *)malloc(sizeof(uint32_t) * obj->mask_size);
        buffer = (void *)raw_mask;
    } else {
        raw_mask_64 = (uint64_t *)malloc(sizeof(uint64_t) * obj->mask_size);
        buffer = (void *)raw_mask_64;
    }

    if (H5Dread(mask_dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer) < 0) {
        fprintf(stderr, "Error: While reading mask\n");
        exit(1);
    }

    // count 0's

    size_t zero = 0;

    obj->mask = (uint8_t *)malloc(sizeof(uint8_t) * obj->mask_size);

    if (mask_dsize == 4) {
        for (size_t j = 0; j < obj->mask_size; j++) {
            if (raw_mask[j] == 0) {
                zero++;
                obj->mask[j] = 1;
            } else {
                obj->mask[j] = 0;
            }
        }
    } else {
        for (size_t j = 0; j < obj->mask_size; j++) {
            if (raw_mask_64[j] == 0) {
                zero++;
                obj->mask[j] = 1;
            } else {
                obj->mask[j] = 0;
            }
        }
    }

    // blit mask over to module mask

    size_t fast, slow, offset, target, image_slow, image_fast, module_pixels;
    module_pixels = E2XE_MOD_FAST * E2XE_MOD_SLOW;

    if (obj->mask_size == E2XE_16M_SLOW * E2XE_16M_FAST) {
        slow = 8;
        fast = 4;
        image_slow = E2XE_16M_SLOW;
        image_fast = E2XE_16M_FAST;
    } else {
        slow = 4;
        fast = 2;
        image_slow = E2XE_4M_SLOW;
        image_fast = E2XE_4M_FAST;
    }
    obj->module_mask = (uint8_t *)malloc(sizeof(uint8_t) * fast * slow * module_pixels);
    for (size_t _slow = 0; _slow < slow; _slow++) {
        size_t row0 = _slow * (E2XE_MOD_SLOW + E2XE_GAP_SLOW) * image_fast;
        for (size_t _fast = 0; _fast < fast; _fast++) {
            for (size_t row = 0; row < E2XE_MOD_SLOW; row++) {
                offset =
                  (row0 + row * image_fast + _fast * (E2XE_MOD_FAST + E2XE_GAP_FAST));
                target = (_slow * fast + _fast) * module_pixels + row * E2XE_MOD_FAST;
                memcpy((void *)&obj->module_mask[target],
                       (void *)&obj->mask[offset],
                       sizeof(uint8_t) * E2XE_MOD_FAST);
            }
        }
    }

    printf("%ld of the pixels are valid\n", zero);

    // cleanup

    if (raw_mask) free(raw_mask);
    if (raw_mask_64) free(raw_mask_64);
    H5Dclose(mask_dataset);
}

/// Get number of VDS and read info about all the sub-files.
///
/// @param master           HDF5 File object pointing to the master file
/// @param dataset          The root dataset to search for VDS from
/// @param data_files_array Pointer to an array variable, that will be
///                         allocated and filled with basic information
///                         about the VDS sub-files.
/// @returns The number of VDS found and allocated into data_files_array
int vds_info(char *root, hid_t master, hid_t dataset, h5_data_file **data_files_array) {
    hid_t plist, vds_source;
    size_t vds_count;
    herr_t status;

    plist = H5Dget_create_plist(dataset);

    status = H5Pget_virtual_count(plist, &vds_count);

    *data_files_array = calloc(vds_count, sizeof(h5_data_file));
    // Used to use vds parameter directly - put here so no mass-changes
    h5_data_file *vds = *data_files_array;

    for (int j = 0; j < vds_count; j++) {
        hsize_t start[MAXDIM], stride[MAXDIM], count[MAXDIM], block[MAXDIM];
        size_t dims;

        vds_source = H5Pget_virtual_vspace(plist, j);
        dims = H5Sget_simple_extent_ndims(vds_source);

        if (dims != 3) {
            H5Sclose(vds_source);
            fprintf(stderr, "incorrect data dimensionality: %d\n", (int)dims);
            return -1;
        }

        H5Sget_regular_hyperslab(vds_source, start, stride, count, block);
        H5Sclose(vds_source);

        H5Pget_virtual_filename(plist, j, vds[j].filename, MAXFILENAME);
        H5Pget_virtual_dsetname(plist, j, vds[j].dsetname, MAXFILENAME);

        for (int k = 1; k < dims; k++) {
            if (start[k] != 0) {
                fprintf(stderr, "incorrect chunk start: %d\n", (int)start[k]);
                return -1;
            }
        }

        vds[j].frames = block[0];
        vds[j].offset = start[0];

        if ((strlen(vds[j].filename) == 1) && (vds[j].filename[0] == '.')) {
            H5L_info_t info;
            status = H5Lget_info(master, vds[j].dsetname, &info, H5P_DEFAULT);

            if (status) {
                fprintf(stderr, "error from H5Lget_info on %s\n", vds[j].dsetname);
                return -1;
            }

            /* if the data file points to an external source, dereference */

            if (info.type == H5L_TYPE_EXTERNAL) {
                char buffer[MAXFILENAME], scr[MAXFILENAME];
                unsigned flags;
                const char *nameptr, *dsetptr;

                H5Lget_val(master, vds[j].dsetname, buffer, MAXFILENAME, H5P_DEFAULT);
                H5Lunpack_elink_val(
                  buffer, info.u.val_size, &flags, &nameptr, &dsetptr);

                /* assumptions herein:
                    - external link references are local paths
                    - only need to worry about UNIX paths e.g. pathsep is /
                    - ASCII so chars are ... chars
                   so manually assemble...
                 */

                strcpy(scr, root);
                scr[strlen(root)] = '/';
                strcpy(scr + strlen(root) + 1, nameptr);

                strcpy(vds[j].filename, scr);
                strcpy(vds[j].dsetname, dsetptr);
            }
        } else {
            char scr[MAXFILENAME];
            sprintf(scr, "%s/%s", root, vds[j].filename);
            strcpy(vds[j].filename, scr);
        }

        // do I want to open these here? Or when they are needed...
        vds[j].file = 0;
        vds[j].dataset = 0;
    }

    status = H5Pclose(plist);

    return vds_count;
}

/// Extracts the h5_data_file dictionary for information on all VDS
///
/// @param filename         The name of the master file
/// @param h5_data_file     The data_files array to be allocated and filled
///
/// @returns The number of VDS files
int unpack_vds(const char *filename, h5_data_file **data_files) {
    // TODO if we want this to become SWMR aware in the future will need to
    // allow for that here
    hid_t file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);

    if (file < 0) {
        fprintf(stderr, "Error: Opening for VDS read %s\n", filename);
        return -1;
    }

    hid_t dataset = H5Dopen(file, "/entry/data/data", H5P_DEFAULT);
    if (dataset < 0) {
        H5Fclose(file);
        fprintf(stderr, "Error: Reading H5 entry %s\n", "/entry/data/data");
        return -1;
    }

    /* always set the absolute path to file information */
    char rootpath[MAXFILENAME];
    strncpy(rootpath, filename, MAXFILENAME);
    char *root = dirname(rootpath);
    char cwd[MAXFILENAME];
    if ((strlen(root) == 1) && (root[0] == '.')) {
        root = getcwd(cwd, MAXFILENAME);
    }

    int vds_count = vds_info(root, file, dataset, data_files);

    H5Dclose(dataset);
    H5Fclose(file);

    return vds_count;
}

void setup_data(h5read_handle *obj) {
    hid_t dataset = obj->data_files[0].dataset;
    hid_t datatype = H5Dget_type(dataset);

    if (H5Tget_size(datatype) != 2) {
        fprintf(stderr, "native data size != 2 (%ld)\n", H5Tget_size(datatype));
        exit(1);
    }

    hid_t space = H5Dget_space(dataset);

    if (H5Sget_simple_extent_ndims(space) != 3) {
        fprintf(stderr, "raw data not three dimensional\n");
        exit(1);
    }

    hsize_t dims[3];
    H5Sget_simple_extent_dims(space, dims, NULL);

    obj->slow = dims[1];
    obj->fast = dims[2];

    printf("Total data size: %ldx%ldx%ld\n", obj->frames, obj->slow, obj->fast);
    H5Sclose(space);
}

h5read_handle *h5read_open(const char *master_filename) {
    hid_t master_file = H5Fopen(master_filename, H5F_ACC_RDONLY, H5P_DEFAULT);

    if (master_file < 0) {
        fprintf(stderr, "Error: Reading %s\n", master_filename);
        return NULL;
    }

    // Create the H5 handle object
    h5read_handle *file = calloc(1, sizeof(h5read_handle));
    file->master_file = master_file;

    file->data_file_count = unpack_vds(master_filename, &file->data_files);

    if (file->data_file_count < 0) {
        fprintf(stderr, "Error: While reading VDS of %s\n", master_filename);
        H5Fclose(master_file);
        free(file);
        return NULL;
    }

    // open up the actual data files, count all the frames
    file->frames = 0;
    h5_data_file *data_files = file->data_files;
    for (int j = 0; j < file->data_file_count; j++) {
        data_files[j].file =
          H5Fopen(data_files[j].filename, H5F_ACC_RDONLY, H5P_DEFAULT);
        if (data_files[j].file < 0) {
            fprintf(stderr, "Error: Opening child file %s\n", data_files[j].filename);
            // Lots of code to cleanup here, so just quit for now
            exit(1);
        }
        data_files[j].dataset =
          H5Dopen(data_files[j].file, data_files[j].dsetname, H5P_DEFAULT);
        if (data_files[j].dataset < 0) {
            fprintf(stderr,
                    "Error: Reading datasets of child file %s\n",
                    data_files[j].filename);
            // Lots of code to cleanup here, so just quit for now
            exit(1);
        }
        file->frames += data_files[j].frames;
    }

    read_mask(file);

    setup_data(file);

    return file;
}

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
    if (!verbose) {
        // Turn off verbose hdf5 errors
        H5Eset_auto(H5E_DEFAULT, NULL, NULL);
    }
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
        handle = h5read_open(argv[1]);
    }
    if (handle == NULL) {
        fprintf(stderr, "Error: Could not open nexus file %s\n", argv[1]);
        exit(1);
    }
    return handle;
}

void calculate_mask_kernels(h5read_handle *obj, int* mask_kernels) {
    size_t ysize = obj->slow;
    size_t xsize = obj->fast;
    int kxsize = 3;
    int kysize = 3;

    int *mask_sat = malloc(sizeof(int) * obj->slow * obj->fast);

    for (size_t j = 0, k = 0; j < ysize; ++j) {
        int m = 0;
        for (size_t i = 0; i < xsize; ++i, ++k) {
            int mm = (obj->mask[k]) ? 1 : 0;
            m += mm;
            if (j == 0) {
                mask_sat[k] = m;
            } else {
                mask_sat[k] = mask_sat[k - xsize] + m;
            }
        }
    }

    for (size_t j = 0; j < ysize; ++j) {
        for (size_t i = 0; i < xsize; ++i) {
            size_t k = j * xsize + i;

            int i0 = i - kxsize - 1, i1 = i + kxsize;
            int j0 = j - kysize - 1, j1 = j + kysize;
            i1 = i1 < xsize ? i1 : xsize - 1;
            j1 = j1 < ysize ? j1 : ysize - 1;
            int k0 = j0 * xsize;
            int k1 = j1 * xsize;

            // Compute the number of points valid in the local area.
            int m = 0;
            if (i0 >= 0 && j0 >= 0) {
                int d00 = mask_sat[k0 + i0];
                int d10 = mask_sat[k1 + i0];
                int d01 = mask_sat[k0 + i1];
                m += d00 - (d10 + d01);
            } else if (i0 >= 0) {
                int  d10 = mask_sat[k1 + i0];
                m -= d10;
            } else if (j0 >= 0) {
                int  d01 = mask_sat[k0 + i1];
                m -= d01;
            }
            int d11 = mask_sat[k1 + i1];
            m += d11;
            mask_kernels[k] = m;
        }
    }

    free(mask_sat);
}

h5read_handle_precalc *h5read_parse_standard_args_precalc(int argc, char **argv) {
    h5read_handle_precalc *new_handle = calloc(1, sizeof(h5read_handle_precalc));
    // h5read_handle *old_handle = h5read_parse_standard_args(argc, argv);
    new_handle->obj = h5read_parse_standard_args(argc, argv);
    // old_handle;
    new_handle->mask_kernels = malloc(sizeof(int)*(new_handle->obj->slow)*(new_handle->obj->fast));
    calculate_mask_kernels(new_handle->obj, new_handle->mask_kernels);
    return new_handle;
}

h5read_handle_precalc *h5read_precalc(h5read_handle *obj) {
    h5read_handle_precalc *new_handle = (h5read_handle_precalc*)calloc(1, sizeof(h5read_handle_precalc));
    new_handle->obj = obj;
    new_handle->mask_kernels = (int*)malloc(sizeof(int)*(new_handle->obj->slow)*(new_handle->obj->fast));
    calculate_mask_kernels(new_handle->obj, new_handle->mask_kernels);
    return new_handle;
}