#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "jpg_utils.cuh"
#include "model_utils.cuh"
#include "bin_utils.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <windows.h>
#include <float.h>

// Function prototypes for different mains
void test_convolution();
void test_convolution_with_weights();
void test_max_pooling();
void test_average_pooling();
void test_residual_connection();
void test_fully_connected();
void test_relu();
void test_batch_normalization();
void test_read_conv_weights();
void test_read_linear();
void test_read_batch_normalization();
void test_read_image();
void classify_image();

int main() {
    classify_image();  // Change this to switch the entry point
    return 0;
}

void test_convolution() {
    printf("Test convolution: \n\n");

    // Variabiles to store the clock cicles used to mesure the execution time
    time_t start;
    time_t stop;
    double elapsed_time;

    // Set the parameters
    int image_size = 6;
    int num_channels = 128;
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

    // Convolution in serial    
    struct tensor output_tensor_serial;
    output_tensor_serial.col = (img_tensor.col + stride - 1) / stride;
    output_tensor_serial.row = (img_tensor.row + stride - 1) / stride;
    output_tensor_serial.depth = num_filters;
    output_tensor_serial.data = (float*)malloc(output_tensor_serial.row * output_tensor_serial.col * output_tensor_serial.depth * sizeof(float));

    // Fill the output data with zeros
    memset(output_tensor_serial.data, 0, output_tensor_serial.row * output_tensor_serial.col * output_tensor_serial.depth * sizeof(float));

    start = clock();
    // Perform convolution in serial
    convolution_serial(&img_tensor, kernels, num_filters, stride, &output_tensor_serial);
    stop = clock();
    elapsed_time = ((double)stop - start) / CLOCKS_PER_SEC;
    printf("Convolution in serial takes: %lf [s]\n", elapsed_time);

    // Convolution in parallel
    // Declare the structure to store the output of the convolution with GPU
    struct tensor output_tensor_parallel;
    output_tensor_parallel.col = (img_tensor.col + stride - 1) / stride;
    output_tensor_parallel.row = (img_tensor.row + stride - 1) / stride;
    output_tensor_parallel.depth = num_filters;
    output_tensor_parallel.data = (float*)malloc(output_tensor_parallel.row * output_tensor_parallel.col * output_tensor_parallel.depth * sizeof(float));

    // Fill the output data with zeros
    memset(output_tensor_parallel.data, 0, output_tensor_parallel.row * output_tensor_parallel.col * output_tensor_parallel.depth * sizeof(float));

    start = clock();
    // Perform convolution using GPU
    cudaError_t cudaStatus = Conv2dWithCuda(&img_tensor, kernels, kernel_size, stride, num_filters, &output_tensor_parallel);
    stop = clock();
    elapsed_time = ((double)stop - start) / CLOCKS_PER_SEC;
    printf("Convolution in parallel takes: %lf [s]\n", elapsed_time);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Conv2dWithCuda failed!");
    }

    // Check the results
    bool correct = true;
    int num_output_data = output_tensor_parallel.depth * output_tensor_parallel.row * output_tensor_parallel.col;
    for (int i = 0; i < num_output_data; i++) {
        if (output_tensor_parallel.data[i] != output_tensor_serial.data[i]) {
            correct = false;
        }
    }

    if (correct) {
        printf("Parallelizzation COMPLETE succesfully!\n");
    }
    else {
        printf("Matrices are NOT equal!\n");      
    }

    // Free the image tensor memory
    free_tensor(&img_tensor);
    for (int i = 0; i < num_filters; i++) {
        free_tensor(&kernels[i]);
    }
    free_tensor(&output_tensor_serial);
    free_tensor(&output_tensor_parallel);
}

void test_convolution_with_weights() {
    printf("Test Conv2dWithCuda: \n\n");

    // Variabiles to store the clock cicles used to mesure the execution time
    time_t start;
    time_t stop;
    double elapsed_time;

    // Set the parameters
    int image_size = 56;
    int num_channels = 64;
    int kernel_size = 1;
    int num_filters = 128;
    int stride = 2;

    // Read the image and store it in a tensor
    struct tensor img_tensor;
    img_tensor.row = image_size;
    img_tensor.col = image_size;
    img_tensor.depth = num_channels;
    img_tensor.data = (float*)malloc(img_tensor.row * img_tensor.col * img_tensor.depth * sizeof(float));
    init_tensor(&img_tensor);

    // Declare the kernel
    const char* filename = "./../../../Parameters/conv_weights_19.bin";

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

    float max = FLT_MIN;
    float min = FLT_MAX;
    for (int i = 0; i < 28 * 28 * 128; i++) {
        if (output_tensor.data[i] > max) {
            max = output_tensor.data[i];
        }
        if (output_tensor.data[i] < min) {
            min = output_tensor.data[i];
        }
    }
    printf("Output layer 1:\n");
    printf("Max value: %f\n", max);
    printf("Min value: %f\n", min);
    
    // Free the image tensor memory
    free_tensor(&img_tensor);
    for (int i = 0; i < num_filters; i++) {
        free_tensor(&kernels[i]);
    }
    free_tensor(&output_tensor);
}

void test_max_pooling() {
    printf("Test max pooling: \n\n");

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

    // Max Pooling in serial
    // Declare the structure to store the output of the convolution with GPU
    struct tensor output_tensor_serial;
    output_tensor_serial.col = (img_tensor.col + stride - 1) / stride;
    output_tensor_serial.row = (img_tensor.row + stride - 1) / stride;
    output_tensor_serial.depth = img_tensor.depth;
    output_tensor_serial.data = (float*)malloc(output_tensor_serial.row * output_tensor_serial.col * output_tensor_serial.depth * sizeof(float));

    // Fill the output data with zeros
    memset(output_tensor_serial.data, 0, output_tensor_serial.row * output_tensor_serial.col * output_tensor_serial.depth * sizeof(float));

    start = clock();
    // Perform max pooling in serial
    max_pooling_serial(&img_tensor, pool_size, stride, &output_tensor_serial);
    stop = clock();
    elapsed_time = ((double)stop - start) / CLOCKS_PER_SEC;
    printf("Convolution in serial takes: %lf [s]\n", elapsed_time);

    // Max Pooling in parallel
    // Declare the structure to store the output of the convolution with GPU
    struct tensor output_tensor_parallel;
    output_tensor_parallel.col = (img_tensor.col + stride - 1) / stride;
    output_tensor_parallel.row = (img_tensor.row + stride - 1) / stride;
    output_tensor_parallel.depth = img_tensor.depth;
    output_tensor_parallel.data = (float*)malloc(output_tensor_parallel.row * output_tensor_parallel.col * output_tensor_parallel.depth * sizeof(float));

    // Fill the output data with zeros
    memset(output_tensor_parallel.data, 0, output_tensor_parallel.row * output_tensor_parallel.col * output_tensor_parallel.depth * sizeof(float));

    start = clock();
    // Perform convolution using GPU
    cudaError_t cudaStatus = MaxPoolingWithCuda(&img_tensor, pool_size, stride, &output_tensor_parallel);
    stop = clock();
    elapsed_time = ((double)stop - start) / CLOCKS_PER_SEC;
    printf("Convolution in parallel takes: %lf [s]\n", elapsed_time);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "MaxPoolingWithCuda failed!");
    }

    // Check the results
    bool correct = true;
    int num_output_data = output_tensor_parallel.depth * output_tensor_parallel.row * output_tensor_parallel.col;
    for (int i = 0; i < num_output_data; i++) {
        if (output_tensor_parallel.data[i] != output_tensor_serial.data[i]) {
            correct = false;
        }
    }

    if (correct) {
        printf("Parallelizzation COMPLETE succesfully!\n");
    }
    else {
        printf("Matrices are NOT equal!\n");
    }

    // Free the image tensor memory
    free_tensor(&img_tensor);
    free_tensor(&output_tensor_serial);
    free_tensor(&output_tensor_parallel);
}

void test_average_pooling() {
    printf("Test average pooling: \n\n");

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

    // Average Pooling in serial
   // Declare the structure to store the output of the convolution with GPU
    float* output_array_serial = (float*)malloc(num_channels * sizeof(float));

    start = clock();
    // Perform max pooling in serial
    average_pooling_serial(&img_tensor, output_array_serial);
    stop = clock();
    elapsed_time = ((double)stop - start) / CLOCKS_PER_SEC;
    printf("Average pooling in serial takes: %lf [s]\n", elapsed_time);

    // Average Pooling in parallel
    // Declare the structure to store the output of the convolution with GPU
    float* output_array_parallel = (float*)malloc(num_channels * sizeof(float));

    start = clock();
    // Perform convolution using GPU
    cudaError_t cudaStatus = AveragePoolingWithCuda(&img_tensor, output_array_parallel);
    stop = clock();
    elapsed_time = ((double)stop - start) / CLOCKS_PER_SEC;
    printf("Average pooling in parallel takes: %lf [s]\n", elapsed_time);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "AveragePoolingWithCuda failed!");
    }

    // Check the results
    bool correct = true;
    int num_output_data = img_tensor.depth;
    for (int i = 0; i < num_output_data; i++) {
        if (output_array_parallel[i] != output_array_serial[i]) {
            correct = false;
        }
    }

    if (correct) {
        printf("Parallelizzation COMPLETE succesfully!\n");
    }
    else {
        printf("Matrices are NOT equal!\n");
    }

    // Free the image tensor memory
    free_tensor(&img_tensor);
    free(output_array_serial);
    free(output_array_parallel);
}

void test_residual_connection() {
    printf("Test residual connection: \n\n");

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

    // Residual connection in serial
    // Declare the structure to store the output of the convolution with GPU
    struct tensor output_tensor_serial;
    output_tensor_serial.col = image_size;
    output_tensor_serial.row = image_size;
    output_tensor_serial.depth = num_channels;
    output_tensor_serial.data = (float*)malloc(output_tensor_serial.row * output_tensor_serial.col * output_tensor_serial.depth * sizeof(float));

    // Fill the output data with zeros
    memset(output_tensor_serial.data, 0, output_tensor_serial.row * output_tensor_serial.col * output_tensor_serial.depth * sizeof(float));

    start = clock();
    // Perform max pooling in serial
    add_tensors_serial(&img_tensor1, &img_tensor2, &output_tensor_serial);
    stop = clock();
    elapsed_time = ((double)stop - start) / CLOCKS_PER_SEC;
    printf("Residual connection in serial takes: %lf [s]\n", elapsed_time);

    // Residual connection in parallel
    // Declare the structure to store the output of the convolution with GPU
    struct tensor output_tensor_parallel;
    output_tensor_parallel.col = image_size;
    output_tensor_parallel.row = image_size;
    output_tensor_parallel.depth = num_channels;
    output_tensor_parallel.data = (float*)malloc(output_tensor_parallel.row * output_tensor_parallel.col * output_tensor_parallel.depth * sizeof(float));

    // Fill the output data with zeros
    memset(output_tensor_parallel.data, 0, output_tensor_parallel.row * output_tensor_parallel.col * output_tensor_parallel.depth * sizeof(float));

    start = clock();
    // Perform convolution using GPU
    cudaError_t cudaStatus = ResidualConnectionWithCuda(&img_tensor1, &img_tensor2, &output_tensor_parallel);
    stop = clock();
    elapsed_time = ((double)stop - start) / CLOCKS_PER_SEC;
    printf("Residual connection in parallel takes: %lf [s]\n", elapsed_time);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ResidualConnectionWithCuda failed!");
    }

    // Check the results
    bool correct = true;
    int num_output_data = output_tensor_parallel.depth * output_tensor_parallel.row * output_tensor_parallel.col;
    for (int i = 0; i < num_output_data; i++) {
        if (output_tensor_parallel.data[i] != output_tensor_serial.data[i]) {
            correct = false;
        }
    }

    if (correct) {
        printf("Parallelizzation COMPLETE succesfully!\n");
    }
    else {
        printf("Matrices are NOT equal!\n");
    }

    // Free the image tensor memory
    free_tensor(&img_tensor1);
    free_tensor(&img_tensor2);
    free_tensor(&output_tensor_serial);
    free_tensor(&output_tensor_parallel);
}

void test_fully_connected() {
    printf("Test fully connected layer: \n\n");

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

    // Fully connected in serial
   // Declare the structure to store the output of the convolution with GPU
    float* output_array_serial = (float*)malloc(num_classes * sizeof(float));

    start = clock();
    // Perform max pooling in serial
    fully_connected_serial(input_array, weights, bias, array_size, num_classes, output_array_serial);
    stop = clock();
    elapsed_time = ((double)stop - start) / CLOCKS_PER_SEC;
    printf("Average pooling in serial takes: %lf [s]\n", elapsed_time);

    // Fully connected in parallel
    // Declare the structure to store the output of the convolution with GPU
    float* output_array_parallel = (float*)malloc(num_classes * sizeof(float));

    start = clock();
    // Perform convolution using GPU
    cudaError_t cudaStatus = FullyConnectedLayerWithCuda(input_array, weights, bias, array_size, num_classes, output_array_parallel);
    stop = clock();
    elapsed_time = ((double)stop - start) / CLOCKS_PER_SEC;
    printf("Fully connected layer in parallel takes: %lf [s]\n", elapsed_time);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "FullyConnectedLayerWithCuda failed!");
    }

    // Check the results
    bool correct = true;
    for (int i = 0; i < num_classes; i++) {
        if (output_array_parallel[i] != output_array_serial[i]) {
            correct = false;
        }
    }

    if (correct) {
        printf("Parallelizzation COMPLETE succesfully!\n");
    }
    else {
        printf("Matrices are NOT equal!\n");
    }

    // Free the memory
    free(input_array);
    free(weights);
    free(bias);
    free(output_array_serial);
    free(output_array_parallel);
}

void test_relu() {
    printf("Test ReLU: \n\n");

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

    // Residual connection in serial
    // Declare the structure to store the output of the convolution with GPU
    struct tensor output_tensor_serial;
    output_tensor_serial.col = image_size;
    output_tensor_serial.row = image_size;
    output_tensor_serial.depth = num_channels;
    output_tensor_serial.data = (float*)malloc(output_tensor_serial.row * output_tensor_serial.col * output_tensor_serial.depth * sizeof(float));

    // Fill the output data with zeros
    memset(output_tensor_serial.data, 0, output_tensor_serial.row * output_tensor_serial.col * output_tensor_serial.depth * sizeof(float));

    start = clock();
    // Perform max pooling in serial
    relu_serial(&img_tensor, &output_tensor_serial);
    stop = clock();
    elapsed_time = ((double)stop - start) / CLOCKS_PER_SEC;
    printf("ReLU in serial takes: %lf [s]\n", elapsed_time);

    // ReLU in parallel
    // Declare the structure to store the output of the convolution with GPU
    struct tensor output_tensor_parallel;
    output_tensor_parallel.col = image_size;
    output_tensor_parallel.row = image_size;
    output_tensor_parallel.depth = num_channels;
    output_tensor_parallel.data = (float*)malloc(output_tensor_parallel.row * output_tensor_parallel.col * output_tensor_parallel.depth * sizeof(float));

    // Fill the output data with zeros
    memset(output_tensor_parallel.data, 0, output_tensor_parallel.row * output_tensor_parallel.col * output_tensor_parallel.depth * sizeof(float));

    start = clock();
    // Perform convolution using GPU
    cudaError_t cudaStatus = ReLUWithCuda(&img_tensor, &output_tensor_parallel);
    stop = clock();
    elapsed_time = ((double)stop - start) / CLOCKS_PER_SEC;
    printf("ReLU in parallel takes: %lf [s]\n", elapsed_time);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ReLUWithCuda failed!");
    }

    // Check the results
    bool correct = true;
    int num_output_data = output_tensor_parallel.depth * output_tensor_parallel.row * output_tensor_parallel.col;
    for (int i = 0; i < num_output_data; i++) {
        if (output_tensor_parallel.data[i] != output_tensor_serial.data[i]) {
            correct = false;
        }
    }

    if (correct) {
        printf("Parallelizzation COMPLETE succesfully!\n");
    }
    else {
        printf("Matrices are NOT equal!\n");
    }

    // Free the image tensor memory
    free_tensor(&img_tensor);
    free_tensor(&output_tensor_parallel);
}

void test_batch_normalization() {
    printf("Test batch normalization: \n\n");

    // Variabiles to store the clock cicles used to mesure the execution time
    time_t start;
    time_t stop;
    double elapsed_time;

    // Set the parameters
    int image_size = 56;
    int num_channels = 64;
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
    load_array("./../../../Parameters/batch_beta_8.bin", conv1_batch1_beta, num_filters);
    conv1_batch1_gamma = (float*)malloc(num_filters * sizeof(float));
    load_array("./../../../Parameters/batch_gamma_8.bin", conv1_batch1_gamma, num_filters);
    conv1_batch1_mean = (float*)malloc(num_filters * sizeof(float));
    load_array("./../../../Parameters/batch_mean_8.bin", conv1_batch1_mean, num_filters);
    conv1_batch1_std = (float*)malloc(num_filters * sizeof(float));
    load_array("./../../../Parameters/batch_std_8.bin", conv1_batch1_std, num_filters);

    // Batch normalization in serial
    // Declare the structure to store the output of the convolution with GPU
    struct tensor output_tensor_serial;
    output_tensor_serial.col = image_size;
    output_tensor_serial.row = image_size;
    output_tensor_serial.depth = num_channels;
    output_tensor_serial.data = (float*)malloc(output_tensor_serial.row * output_tensor_serial.col * output_tensor_serial.depth * sizeof(float));

    // Fill the output data with zeros
    memset(output_tensor_serial.data, 0, output_tensor_serial.row * output_tensor_serial.col * output_tensor_serial.depth * sizeof(float));

    start = clock();
    // Perform max pooling in serial
    batch_normalization_serial(&img_tensor, conv1_batch1_beta, conv1_batch1_gamma, conv1_batch1_mean, conv1_batch1_std, &output_tensor_serial);
    stop = clock();
    elapsed_time = ((double)stop - start) / CLOCKS_PER_SEC;
    printf("Batch normalization in serial takes: %lf [s]\n", elapsed_time);

    // Batch normalization in parallel
    // Declare the structure to store the output of the convolution with GPU
    struct tensor output_tensor_parallel;
    output_tensor_parallel.col = image_size;
    output_tensor_parallel.row = image_size;
    output_tensor_parallel.depth = num_channels;
    output_tensor_parallel.data = (float*)malloc(output_tensor_parallel.row * output_tensor_parallel.col * output_tensor_parallel.depth * sizeof(float));

    // Fill the output data with zeros
    memset(output_tensor_parallel.data, 0, output_tensor_parallel.row * output_tensor_parallel.col * output_tensor_parallel.depth * sizeof(float));

    start = clock();
    // Perform convolution using GPU
    cudaError_t cudaStatus = BatchNormalizationWithCuda(&img_tensor, conv1_batch1_beta, conv1_batch1_gamma, conv1_batch1_mean, conv1_batch1_std, &output_tensor_parallel);
    stop = clock();
    elapsed_time = ((double)stop - start) / CLOCKS_PER_SEC;
    printf("Batch normalization in parallel takes: %lf [s]\n", elapsed_time);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "BatchNormalizationWithCuda failed!");
    }

    // Check the results
    bool correct = true;
    int num_output_data = output_tensor_parallel.depth * output_tensor_parallel.row * output_tensor_parallel.col;
    for (int i = 0; i < num_output_data; i++) {
        if (output_tensor_parallel.data[i] != output_tensor_serial.data[i]) {
            correct = false;
        }
    }

    if (correct) {
        printf("Parallelizzation COMPLETE succesfully!\n");
    }
    else {
        printf("Matrices are NOT equal!\n");
    }

    // Free the image tensor memory
    free_tensor(&img_tensor);
    free_tensor(&output_tensor_parallel);
}

void test_read_conv_weights() {
    // File to read
    const char* filename = "./../../../Parameters/conv_weights_4.bin";

    // Define the size of the kernel
    int kernel_size = 3;
    int num_channels = 64;
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
    const char* filename = "./../../../Parameters/batch_mean_46.bin";

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

void test_read_image() {
    // File to read
    char* filename = "./../../../dog.bin";

    // Read the image and store it in a tensor
    struct tensor img_tensor;
    img_tensor.row = 224;
    img_tensor.col = 224;
    img_tensor.depth = 3;
    img_tensor.data = (float*)malloc(img_tensor.depth  * img_tensor.row  * img_tensor.col * sizeof(float));

    load_image_as_tensor(filename, &img_tensor);

    // Check the results
    print_tensor(&img_tensor);

    free_tensor(&img_tensor);
}

void classify_image() {

    // Check the working directory
    char cwd[1000];
    if (GetCurrentDirectory(sizeof(cwd), cwd) != 0) {
        printf("Current working dir: %s\n", cwd);
    }
    else {
        printf("GetCurrentDirectory() error: %lu\n", GetLastError());
    }

    printf("Classify image dog.bin: \n");

    // Variabiles to store the clock cicles used to mesure the execution time
    time_t start;
    time_t stop;
    double elapsed_time;

    // File to read
    char* filename = "./../../../dog.bin";

    // Set the parameters
    int image_size = 224;
    int num_channels = 3;
    int num_classes = 1000;

    // Read the image and store it in a tensor
    struct tensor img_tensor;
    img_tensor.row = image_size;
    img_tensor.col = image_size;
    img_tensor.depth = num_channels;
    img_tensor.data = (float*)malloc(img_tensor.depth * img_tensor.row * img_tensor.col * sizeof(float));

    load_image_as_tensor(filename, &img_tensor);

    printf("Image read succesfully\n");
    printf("Image size: %d, %d, %d\n", img_tensor.depth, img_tensor.row, img_tensor.col);

    // Read the classes
    filename = "./../../../imagenet_classes.txt";

    char** classes = load_classes(filename, num_classes);
    if (classes == NULL) {
        exit(-1);
    }

    // ResNet in serial
    // Declare the structure to store the output of the convolution with GPU
    float* output_classes_serial;
    output_classes_serial = (float*)malloc(num_classes * sizeof(float));

    memset(output_classes_serial, 0, num_classes * sizeof(float));

    start = clock();
    // Perform ResNet18 in serial
    ResNet_serial(&img_tensor, output_classes_serial);
    stop = clock();
    elapsed_time = ((double)stop - start) / CLOCKS_PER_SEC;
    printf("Convolution in serial takes: %lf [s]\n", elapsed_time);

    // Check the results
    // Array to store the top 5 values and their indices
    int top = 5;
    float* top_prob_serial = (float*)malloc(top * sizeof(float));
    int* top_idx_serial = (int*)malloc(top * sizeof(int));

    // Initialize the top values to the lowest possible float
    for (int i = 0; i < top; i++) {
        top_prob_serial[i] = -FLT_MAX;
        top_idx_serial[i] = -1;
    }

    // Find the top 5 values and their indices
    for (int i = 0; i < num_classes; i++) {
        for (int j = 0; j < top; j++) {
            if (output_classes_serial[i] > top_prob_serial[j]) {
                // Shift the current values down
                for (int k = (top - 1); k > j; k--) {
                    top_prob_serial[k] = top_prob_serial[k - 1];
                    top_idx_serial[k] = top_idx_serial[k - 1];
                }
                // Insert the new value
                top_prob_serial[j] = output_classes_serial[i];
                top_idx_serial[j] = i;
                break;
            }
        }
    }

    // Print the top 5 values and their indices
    printf("Top 5 classes and their probabilities:\n");
    for (int i = 0; i < top; i++) {
        if (top_idx_serial[i] != -1) {
            printf("Class: %s\tValue:%f\n", classes[top_idx_serial[i]], top_prob_serial[i]);
        }
    }

    printf("\n");

    // ResNet (GPU)
    // Declare the structure to store the output of the convolution with GPU
    float* output_classes_parallel;
    output_classes_parallel = (float*)malloc(num_classes * sizeof(float));

    memset(output_classes_parallel, 0, num_classes * sizeof(float));

    start = clock();
    // Perform ResNet18 using GPU
    cudaError_t cudaStatus = ResNetWithCuda(&img_tensor, output_classes_parallel);
    stop = clock();
    elapsed_time = ((double)stop - start) / CLOCKS_PER_SEC;
    printf("Convolution in parallel takes: %lf [s]\n", elapsed_time);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ResNetWithCuda failed!");
    }

    // Check the results
    // Array to store the top 5 values and their indices
    float* top_prob_parallel = (float*)malloc(top * sizeof(float));
    int* top_idx_parallel = (int*)malloc(top * sizeof(int));

    // Initialize the top values to the lowest possible float
    for (int i = 0; i < top; i++) {
        top_prob_parallel[i] = -FLT_MAX;
        top_idx_parallel[i] = -1;
    }

    // Find the top 5 values and their indices
    for (int i = 0; i < num_classes; i++) {
        for (int j = 0; j < top; j++) {
            if (output_classes_parallel[i] > top_prob_parallel[j]) {
                // Shift the current values down
                for (int k = (top - 1); k > j; k--) {
                    top_prob_parallel[k] = top_prob_parallel[k - 1];
                    top_idx_parallel[k] = top_idx_parallel[k - 1];
                }
                // Insert the new value
                top_prob_parallel[j] = output_classes_parallel[i];
                top_idx_parallel[j] = i;
                break;
            }
        }
    }

    // Print the top 5 values and their indices
    printf("Top 5 classes and their probabilities:\n");
    for (int i = 0; i < top; i++) {
        if (top_idx_parallel[i] != -1) {
            printf("Class: %s\tValue:%f\n", classes[top_idx_parallel[i]], top_prob_parallel[i]);
        }
    }

    // Free the image tensor memory
    free_tensor(&img_tensor);
    free(output_classes_serial);
    free(output_classes_parallel);

    // Free the classes memory
    for (int i = 0; i < num_classes; i++) {
        free(classes[i]);  // Free the memory for each string
    }
    free(classes);  // Free the array of pointers
}