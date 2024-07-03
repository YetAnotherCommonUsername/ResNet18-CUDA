#include "bin_utils.cuh"

void load_conv_weights(const char* filename, struct tensor* kernels, int kernel_size, int num_channels, int num_filters) {
    FILE* fin;
    int i, npixel;
    char buffer[200];

    fin = fopen(filename, "rb");
    if (fin == NULL) {
        printf("Error reading the file\n");
        exit(-1);
    }
    // Read the image type.
    // fgets(buffer, sizeof(buffer), fin);

    // Read the file
    int num_params = kernel_size * kernel_size * num_channels;

    for (int i = 0; i < num_filters; i++) {
        kernels[i].row = kernel_size;
        kernels[i].col = kernel_size;
        kernels[i].depth = num_channels;
        kernels[i].data = (float*)malloc(num_params * sizeof(float));

        // Read weights into the kernel's data array
        size_t read_elements = fread(kernels[i].data, sizeof(float), num_params, fin);
        if (read_elements != num_params) {
            printf("Error reading weights from file\n");
            fclose(fin);
            exit(-1);
        }
    }
    fclose(fin);
}

void load_matrix(const char* filename, float* weights, int ncol, int nrow) {
    FILE* fin;
    int i, npixel;
    char buffer[200];

    fin = fopen(filename, "rb");
    if (fin == NULL) {
        printf("Error reading the file\n");
        exit(-1);
    }
    // Read the image type.
    // fgets(buffer, sizeof(buffer), fin);

    // Read the file
    int num_params = ncol * nrow;

    // Read weights into the kernel's data array
    size_t read_elements = fread(weights, sizeof(float), num_params, fin);

    if (read_elements != num_params) {
        printf("Error reading weights from file\n");
        fclose(fin);
        exit(-1);
    }

    fclose(fin);
}

void load_array(const char* filename, float* weights, int size) {
    FILE* fin;
    int i, npixel;
    char buffer[200];

    fin = fopen(filename, "rb");
    if (fin == NULL) {
        printf("Error reading the file\n");
        exit(-1);
    }

    // Read the file
    size_t read_elements = fread(weights, sizeof(float), size, fin);
    
    if (read_elements != size) {
        printf("Error reading weights from file\n");
        fclose(fin);
        exit(-1);
    }

    fclose(fin);
}