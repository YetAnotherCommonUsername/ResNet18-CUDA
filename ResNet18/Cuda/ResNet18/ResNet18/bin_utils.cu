#include "bin_utils.cuh"

#define MAX_LINE_LENGTH 256  // Define the maximum length of each line

void load_conv_weights(const char* filename, struct tensor* kernels, int kernel_size, int num_channels, int num_filters) {
    FILE* fin;

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

char** load_classes(const char* filename, int num_classes) {
    // Open the file for reading
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        return NULL;
    }

    // Allocate memory for an array of strings
    char** strings = (char **)malloc(num_classes * sizeof(char*));
    if (strings == NULL) {
        perror("Error allocating memory");
        fclose(file);
        return NULL;
    }

    // Read each line and store it into the array
    char buffer[MAX_LINE_LENGTH];
    for (int i = 0; i < num_classes; i++) {
        if (fgets(buffer, MAX_LINE_LENGTH, file) == NULL) {
            if (feof(file)) {
                fprintf(stderr, "Error: less than %d lines in file\n", num_classes);
            }
            else {
                perror("Error reading from file");
            }
            // Free allocated memory before returning
            for (int j = 0; j < i; j++) {
                free(strings[j]);
            }
            free(strings);
            fclose(file);
            return NULL;
        }

        // Remove the newline character at the end of the line, if any
        buffer[strcspn(buffer, "\n")] = '\0';

        // Allocate memory for the string and copy it from the buffer
        strings[i] = (char*)malloc((strlen(buffer) + 1) * sizeof(char));
        if (strings[i] == NULL) {
            perror("Error allocating memory for string");
            // Free allocated memory before returning
            for (int j = 0; j < i; j++) {
                free(strings[j]);
            }
            free(strings);
            fclose(file);
            return NULL;
        }
        strcpy(strings[i], buffer);
    }

    // Close the file
    fclose(file);

    return strings;
}

void load_image_as_tensor(const char* filename, struct tensor* img_tensor) {
    FILE* fin;

    fin = fopen(filename, "rb");
    if (fin == NULL) {
        printf("Error reading the file\n");
        exit(-1);
    }

    // Read the file
    int num_data = img_tensor->depth * img_tensor->row * img_tensor->col;

    // Read weights into the kernel's data array
    size_t read_elements = fread(img_tensor->data, sizeof(float), num_data, fin);

    if (read_elements != num_data) {
        printf("Error reading data from image file\n");
        fclose(fin);
        exit(-1);
    }

    fclose(fin);
}
