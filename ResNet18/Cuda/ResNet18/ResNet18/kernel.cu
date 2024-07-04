﻿#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "jpg_utils.cuh"
#include "model_utils.cuh"
#include "bin_utils.cuh"
#include <stdio.h>
#include <time.h>

// Function prototypes for different mains
void test_convolution();
void test_convolution_with_weights();
void test_max_pooling();
void test_average_pooling();
void test_residual_connection();
void test_fully_connected();
void test_relu();
void test_batch_normalization();
void test_read_image();
void test_read_conv_weights();
void test_read_linear();
void test_read_batch_normalization();
void classify_image();

int main() {
    classify_image();  // Change this to switch the entry point
    return 0;
}

void test_convolution() {
    printf("Test Conv2dWithCuda: \n\n");

    // Variabiles to store the clock cicles used to mesure the execution time
    time_t start;
    time_t stop;
    double elapsed_time;

    // Set the parameters
    int image_size = 6;
    int num_channels = 64;
    int kernel_size = 3;
    int num_filters = 2;
    int stride = 1;

    // Read the image and store it in a tensor
    struct tensor img_tensor;
    img_tensor.row = image_size;
    img_tensor.col = image_size;
    img_tensor.depth = num_channels;
    img_tensor.data = (float*)malloc(img_tensor.row * img_tensor.col * img_tensor.depth * sizeof(float));
    init_tensor(&img_tensor);

    // Check img_tensor
    printf("Image tensor:\n");
    print_tensor(&img_tensor);

    // Declare the kernel
    struct tensor* kernels = (struct tensor*)malloc(num_filters * sizeof(struct tensor));
    if (kernels == NULL) {
        fprintf(stderr, "Failed to allocate memory for kernels\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < num_filters; i++) {
        kernels[i].row = kernel_size;
        kernels[i].col = kernel_size;
        kernels[i].depth = num_channels;
        kernels[i].data = (float*)malloc(kernels[i].row * kernels[i].col * kernels[i].depth * sizeof(float));
        init_tensor(&kernels[i]);
    }

    // Check kernels
    printf("Kernel tensors:\n");
    for (int i = 0; i < 2; i++) {
        print_tensor(&kernels[i]);
    }

    // GPU CONVOLUTION
    // Declare the structure to store the output of the convolution with GPU
    struct tensor output_tensor;
    output_tensor.col = (img_tensor.col + stride - 1) / stride;
    output_tensor.row = (img_tensor.row + stride - 1) / stride;
    output_tensor.depth = num_filters;
    output_tensor.data = (float*)malloc(output_tensor.row * output_tensor.col * output_tensor.depth * sizeof(float));

    // Fill the output data with zeros
    memset(output_tensor.data, 0, output_tensor.row * output_tensor.col * output_tensor.depth * sizeof(float));

    start = clock();
    // Perform convolution using GPU
    cudaError_t cudaStatus = Conv2dWithCuda(&img_tensor, kernels, kernel_size, stride, num_filters, &output_tensor);
    stop = clock();
    elapsed_time = ((double)stop - start) / CLOCKS_PER_SEC;
    printf("Convolution in parallel takes: %lf [s]\n", elapsed_time);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Conv2dWithCuda failed!");
    }

    // Check the results
    print_tensor(&output_tensor);

    // Free the image tensor memory
    free_tensor(&img_tensor);
    for (int i = 0; i < num_filters; i++) {
        free_tensor(&kernels[i]);
    }
    free_tensor(&output_tensor);
}

void test_convolution_with_weights() {
    printf("Test Conv2dWithCuda: \n\n");

    // Variabiles to store the clock cicles used to mesure the execution time
    time_t start;
    time_t stop;
    double elapsed_time;

    // Set the parameters
    int image_size = 224;
    int num_channels = 3;
    int kernel_size = 7;
    int num_filters = 64;
    int stride = 2;

    // Read the image and store it in a tensor
    struct tensor img_tensor;
    img_tensor.row = image_size;
    img_tensor.col = image_size;
    img_tensor.depth = num_channels;
    img_tensor.data = (float*)malloc(img_tensor.row * img_tensor.col * img_tensor.depth * sizeof(float));
    init_random_tensor(&img_tensor);

    // Declare the kernel
    const char* filename = "./../../../Parameters/conv_weights_0.bin";

    // Define the kernel tensor
    struct tensor* kernels;
    kernels = (struct tensor*)malloc(num_filters * sizeof(struct tensor));
    load_conv_weights(filename, kernels, kernel_size, num_channels, num_filters);

    print_tensor(&kernels[0]);
    
    // GPU CONVOLUTION
    // Declare the structure to store the output of the convolution with GPU
    struct tensor output_tensor;
    output_tensor.col = (img_tensor.col + stride - 1) / stride;
    output_tensor.row = (img_tensor.row + stride - 1) / stride;
    output_tensor.depth = num_filters;
    output_tensor.data = (float*)malloc(output_tensor.row * output_tensor.col * output_tensor.depth * sizeof(float));

    // Fill the output data with zeros
    memset(output_tensor.data, 0, output_tensor.row * output_tensor.col * output_tensor.depth * sizeof(float));

    start = clock();
    // Perform convolution using GPU
    cudaError_t cudaStatus = Conv2dWithCuda(&img_tensor, kernels, kernel_size, stride, num_filters, &output_tensor);
    stop = clock();
    elapsed_time = ((double)stop - start) / CLOCKS_PER_SEC;
    printf("Convolution in parallel takes: %lf [s]\n", elapsed_time);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Conv2dWithCuda failed!");
    }

    // Check the results
    for (int i = 0; i < 10; i++) {
        printf("%f\n", output_tensor.data[i]);
    }
    
    // Free the image tensor memory
    free_tensor(&img_tensor);
    for (int i = 0; i < num_filters; i++) {
        free_tensor(&kernels[i]);
    }
    free_tensor(&output_tensor);
}

void test_max_pooling() {
    printf("Test MaxPoolingWithCuda: \n\n");

    // Variabiles to store the clock cicles used to mesure the execution time
    time_t start;
    time_t stop;
    double elapsed_time;

    // Set the parameters
    int image_size = 6;
    int num_channels = 3;
    int pool_size = 3;
    int stride = 2;

    // Read the image and store it in a tensor
    struct tensor img_tensor;
    img_tensor.row = image_size;
    img_tensor.col = image_size;
    img_tensor.depth = num_channels;
    img_tensor.data = (float*)malloc(img_tensor.row * img_tensor.col * img_tensor.depth * sizeof(float));
    init_random_tensor(&img_tensor);

    // Check img_tensor
    printf("Image tensor:\n");
    print_tensor(&img_tensor);

    // GPU CONVOLUTION
    // Declare the structure to store the output of the convolution with GPU
    struct tensor output_tensor;
    output_tensor.col = (img_tensor.col + stride - 1) / stride;
    output_tensor.row = (img_tensor.row + stride - 1) / stride;
    output_tensor.depth = img_tensor.depth;
    output_tensor.data = (float*)malloc(output_tensor.row * output_tensor.col * output_tensor.depth * sizeof(float));

    // Fill the output data with zeros
    memset(output_tensor.data, 0, output_tensor.row * output_tensor.col * output_tensor.depth * sizeof(float));

    start = clock();
    // Perform convolution using GPU
    cudaError_t cudaStatus = MaxPoolingWithCuda(&img_tensor, pool_size, stride, &output_tensor);
    stop = clock();
    elapsed_time = ((double)stop - start) / CLOCKS_PER_SEC;
    printf("Convolution in parallel takes: %lf [s]\n", elapsed_time);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "MaxPoolingWithCuda failed!");
    }

    // Check the results
    print_tensor(&output_tensor);

    // Free the image tensor memory
    free_tensor(&img_tensor);
    free_tensor(&output_tensor);
}

void test_average_pooling() {
    printf("Test AveragePoolingWithCuda: \n\n");

    // Variabiles to store the clock cicles used to mesure the execution time
    time_t start;
    time_t stop;
    double elapsed_time;

    // Set the parameters
    int image_size = 6;
    int num_channels = 3;

    // Read the image and store it in a tensor
    struct tensor img_tensor;
    img_tensor.row = image_size;
    img_tensor.col = image_size;
    img_tensor.depth = num_channels;
    img_tensor.data = (float*)malloc(img_tensor.row * img_tensor.col * img_tensor.depth * sizeof(float));
    init_tensor(&img_tensor);

    // Check img_tensor
    printf("Image tensor:\n");
    print_tensor(&img_tensor);

    // GPU CONVOLUTION
    // Declare the structure to store the output of the convolution with GPU
    float* output_array = (float*)malloc(num_channels * sizeof(float));

    start = clock();
    // Perform convolution using GPU
    cudaError_t cudaStatus = AveragePoolingWithCuda(&img_tensor, output_array);
    stop = clock();
    elapsed_time = ((double)stop - start) / CLOCKS_PER_SEC;
    printf("Convolution in parallel takes: %lf [s]\n", elapsed_time);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "AveragePoolingWithCuda failed!");
    }

    // Check the results
    for (int i = 0; i < num_channels; i++) {
        printf("%f\n", output_array[i]);
    }

    // Free the image tensor memory
    free_tensor(&img_tensor);
    free(output_array);
}

void test_residual_connection() {
    printf("Test ResidualConnectionWithCuda: \n\n");

    // Variabiles to store the clock cicles used to mesure the execution time
    time_t start;
    time_t stop;
    double elapsed_time;

    // Set the parameters
    int image_size = 6;
    int num_channels = 3;

    // Read the first image and store it in a tensor
    struct tensor img_tensor1;
    img_tensor1.row = image_size;
    img_tensor1.col = image_size;
    img_tensor1.depth = num_channels;
    img_tensor1.data = (float*)malloc(img_tensor1.row * img_tensor1.col * img_tensor1.depth * sizeof(float));
    init_tensor(&img_tensor1);

    // Read the second image and store it in a tensor
    struct tensor img_tensor2;
    img_tensor2.row = image_size;
    img_tensor2.col = image_size;
    img_tensor2.depth = num_channels;
    img_tensor2.data = (float*)malloc(img_tensor2.row * img_tensor2.col * img_tensor2.depth * sizeof(float));
    init_tensor(&img_tensor2);

    // Check img_tensor
    printf("Image tensor:\n");
    print_tensor(&img_tensor1);

    // GPU CONVOLUTION
    // Declare the structure to store the output of the convolution with GPU
    struct tensor output_tensor;
    output_tensor.col = image_size;
    output_tensor.row = image_size;
    output_tensor.depth = num_channels;
    output_tensor.data = (float*)malloc(output_tensor.row * output_tensor.col * output_tensor.depth * sizeof(float));

    // Fill the output data with zeros
    memset(output_tensor.data, 0, output_tensor.row * output_tensor.col * output_tensor.depth * sizeof(float));

    start = clock();
    // Perform convolution using GPU
    cudaError_t cudaStatus = ResidualConnectionWithCuda(&img_tensor1, &img_tensor2, &output_tensor);
    stop = clock();
    elapsed_time = ((double)stop - start) / CLOCKS_PER_SEC;
    printf("Residual connection in parallel takes: %lf [s]\n", elapsed_time);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ResidualConnectionWithCuda failed!");
    }

    // Check the results
    print_tensor(&output_tensor);

    // Free the image tensor memory
    free_tensor(&img_tensor1);
    free_tensor(&img_tensor2);
    free_tensor(&output_tensor);
}

void test_fully_connected() {
    printf("Test FullyConnectedWithCuda: \n\n");

    // Variabiles to store the clock cicles used to mesure the execution time
    time_t start;
    time_t stop;
    double elapsed_time;

    // Set the parameters
    int array_size = 512;
    int num_classes = 1000;

    // Create the input array
    float* input_array = (float*)malloc(array_size * sizeof(float));
    if (input_array == NULL) {
        fprintf(stderr, "Memory allocation failed for input_array\n");
        return;
    }
    for (int i = 0; i < array_size; i++) {
        input_array[i] = 1.0;
    }

    // Create the weights matrix
    int num_weights = num_classes * array_size;
    float* weights = (float*)malloc(num_weights * sizeof(float));
    if (weights == NULL) {
        fprintf(stderr, "Memory allocation failed for weights\n");
        return;
    }
    for (int i = 0; i < num_weights; i++) {
        weights[i] = 1.0;
    }

    // Create the bias array
    float* bias = (float*)malloc(num_classes * sizeof(float));
    if (bias == NULL) {
        fprintf(stderr, "Memory allocation failed for bias\n");
        return;
    }
    for (int i = 0; i < num_classes; i++) {
        bias[i] = 1.0;
    }

    // GPU CONVOLUTION
    // Declare the structure to store the output of the convolution with GPU
    float* output_array = (float*)malloc(num_classes * sizeof(float));

    start = clock();
    // Perform convolution using GPU
    cudaError_t cudaStatus = FullyConnectedLayerWithCuda(input_array, weights, bias, array_size, num_classes, output_array);
    stop = clock();
    elapsed_time = ((double)stop - start) / CLOCKS_PER_SEC;
    printf("Fully connected layer in parallel takes: %lf [s]\n", elapsed_time);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "FullyConnectedLayerWithCuda failed!");
    }

    // Check the results
    for (int i = 0; i < 10; i++) {
        printf("%f\n", output_array[i]);
    }

    // Free the memory
    free(input_array);
    free(weights);
    free(bias);
    free(output_array);
}

void test_relu() {
    printf("Test ReLUWithCuda: \n\n");

    // Variabiles to store the clock cicles used to mesure the execution time
    time_t start;
    time_t stop;
    double elapsed_time;

    // Set the parameters
    int image_size = 6;
    int num_channels = 3;

    // Read the first image and store it in a tensor
    struct tensor img_tensor;
    img_tensor.row = image_size;
    img_tensor.col = image_size;
    img_tensor.depth = num_channels;
    img_tensor.data = (float*)malloc(img_tensor.row * img_tensor.col * img_tensor.depth * sizeof(float));
    init_random_tensor(&img_tensor);

    // Check img_tensor
    printf("Image tensor:\n");
    print_tensor(&img_tensor);

    // GPU CONVOLUTION
    // Declare the structure to store the output of the convolution with GPU
    struct tensor output_tensor;
    output_tensor.col = image_size;
    output_tensor.row = image_size;
    output_tensor.depth = num_channels;
    output_tensor.data = (float*)malloc(output_tensor.row * output_tensor.col * output_tensor.depth * sizeof(float));

    // Fill the output data with zeros
    memset(output_tensor.data, 0, output_tensor.row * output_tensor.col * output_tensor.depth * sizeof(float));

    start = clock();
    // Perform convolution using GPU
    cudaError_t cudaStatus = ReLUWithCuda(&img_tensor, &output_tensor);
    stop = clock();
    elapsed_time = ((double)stop - start) / CLOCKS_PER_SEC;
    printf("ReLU in parallel takes: %lf [s]\n", elapsed_time);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ReLUWithCuda failed!");
    }

    // Check the results
    print_tensor(&output_tensor);

    // Free the image tensor memory
    free_tensor(&img_tensor);
    free_tensor(&output_tensor);
}

void test_batch_normalization() {
    printf("Test BatchNormalizationWithCuda: \n\n");

    // Variabiles to store the clock cicles used to mesure the execution time
    time_t start;
    time_t stop;
    double elapsed_time;

    // Set the parameters
    int image_size = 224;
    int num_channels = 3;
    int num_filters = 64;

    // Read the first image and store it in a tensor
    struct tensor img_tensor;
    img_tensor.row = image_size;
    img_tensor.col = image_size;
    img_tensor.depth = num_channels;
    img_tensor.data = (float*)malloc(img_tensor.row * img_tensor.col * img_tensor.depth * sizeof(float));
    init_random_tensor(&img_tensor);

    float* conv1_batch1_beta, * conv1_batch1_gamma, * conv1_batch1_mean, * conv1_batch1_std;

    conv1_batch1_beta = (float*)malloc(num_filters * sizeof(float));
    load_array("./../../../Parameters/batch_beta_1.bin", conv1_batch1_beta, num_filters);
    conv1_batch1_gamma = (float*)malloc(num_filters * sizeof(float));
    load_array("./../../../Parameters/batch_gamma_1.bin", conv1_batch1_gamma, num_filters);
    conv1_batch1_mean = (float*)malloc(num_filters * sizeof(float));
    load_array("./../../../Parameters/batch_mean_1.bin", conv1_batch1_mean, num_filters);
    conv1_batch1_std = (float*)malloc(num_filters * sizeof(float));
    load_array("./../../../Parameters/batch_std_1.bin", conv1_batch1_std, num_filters);

    // GPU CONVOLUTION
    // Declare the structure to store the output of the convolution with GPU
    struct tensor output_tensor;
    output_tensor.col = image_size;
    output_tensor.row = image_size;
    output_tensor.depth = num_channels;
    output_tensor.data = (float*)malloc(output_tensor.row * output_tensor.col * output_tensor.depth * sizeof(float));

    // Fill the output data with zeros
    memset(output_tensor.data, 0, output_tensor.row * output_tensor.col * output_tensor.depth * sizeof(float));

    start = clock();
    // Perform convolution using GPU
    cudaError_t cudaStatus = BatchNormalizationWithCuda(&img_tensor, conv1_batch1_beta, conv1_batch1_gamma, conv1_batch1_mean, conv1_batch1_std, &output_tensor);
    stop = clock();
    elapsed_time = ((double)stop - start) / CLOCKS_PER_SEC;
    printf("Batch normalization in parallel takes: %lf [s]\n", elapsed_time);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "BatchNormalizationWithCuda failed!");
    }

    // Check the results
    int n = 10;
    for (int i = 0; i < n; i++) {
        printf("%f\n", output_tensor.data[i]);
    }

    // Free the image tensor memory
    free_tensor(&img_tensor);
    free_tensor(&output_tensor);
}

void test_read_image() {
    struct tensor img_tensor;
    const char* filename = "./../../../dog.jpg";

    load_image_as_tensor(filename, &img_tensor);

    // You can now use img_tensor
    printf("Image loaded: %dx%dx%d\n", img_tensor.row, img_tensor.col, img_tensor.depth);

    // Free allocated memory
    free_tensor(&img_tensor);
}

void test_read_conv_weights() {
    // File to read
    const char* filename = "./../../../Parameters/conv_weights_0.bin";

    // Define the size of the kernel
    int kernel_size = 7;
    int num_channels = 3;
    int num_filters = 64;

    // Define the kernel tensor
    struct tensor* kernels;
    kernels = (struct tensor*)malloc(num_filters * sizeof(struct tensor));

    load_conv_weights(filename, kernels, kernel_size, num_channels, num_filters);

    // Check the results
    print_tensor(&kernels[0]);

    // Free allocated memory
    for (int i = 0; i < num_filters; i++) {
        free_tensor(&kernels[i]);
    }
}

void test_read_linear() {
    // File to read
    const char* filename_weights = "./../../../Parameters/linear_weights_51.bin";
    const char* filename_bias = "./../../../Parameters/linear_bias_51.bin";

    // Define the size of the kernel
    int input_size = 512;
    int num_classes = 1000;

    // Define the weights and bias arrays
    float* weights;
    weights = (float*)malloc(input_size * num_classes * sizeof(float));
    
    float* bias;
    bias = (float*)malloc(num_classes * sizeof(float));


    load_matrix(filename_weights, weights, input_size, num_classes);
    load_array(filename_bias, bias, num_classes);

    // Check the results
    for (int i = 0; i < 10; i++) {
        printf("%f\n", bias[i]);
    }

    // Free allocated memory
    free(weights);
    free(bias);
}

void test_read_batch_normalization() {
    // File to read
    const char* filename = "./../../../Parameters/batch_beta_1.bin";

    // Define the size of the kernel
    int size = 64;

    // Define the weights and bias arrays
    float* bias;
    bias = (float*)malloc(size * sizeof(float));

    load_array(filename, bias, size);

    // Check the results
    for (int i = 0; i < 10; i++) {
        printf("%f\n", bias[i]);
    }

    // Free allocated memory
    free(bias);
}

void classify_image() {
    printf("Classify image: \n\n");

    // Variabiles to store the clock cicles used to mesure the execution time
    time_t start;
    time_t stop;
    double elapsed_time;

    // Set the parameters
    int image_size = 224;
    int num_channels = 3;
    int num_classes = 1000;

    // Read the image and store it in a tensor
    struct tensor img_tensor;
    img_tensor.row = image_size;
    img_tensor.col = image_size;
    img_tensor.depth = num_channels;
    img_tensor.data = (float*)malloc(img_tensor.row * img_tensor.col * img_tensor.depth * sizeof(float));
    init_random_tensor(&img_tensor);

    // GPU CONVOLUTION
    // Declare the structure to store the output of the convolution with GPU
    float* output_classes;
    output_classes = (float*)malloc(num_classes * sizeof(float));

    memset(output_classes, 0, num_classes * sizeof(float));

    start = clock();
    // Perform convolution using GPU
    cudaError_t cudaStatus = ResNetWithCuda(&img_tensor, output_classes);
    stop = clock();
    elapsed_time = ((double)stop - start) / CLOCKS_PER_SEC;
    printf("Convolution in parallel takes: %lf [s]\n", elapsed_time);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ResNetWithCuda failed!");
    }

    /*
    // Check the results
    printf("Logit:");
    for (int i = 0; i < num_classes; i++) {
        if (output_classes[i] != 0.0) {
            printf("Class: %d\tLogit:%f\n", i, output_classes[i]);
        }
    }
    */

    // Free the image tensor memory
    free_tensor(&img_tensor);
    free(output_classes);
}