#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "jpg_utils.cuh"
#include "model_utils.cuh"
#include <stdio.h>
#include <time.h>

int main(int argc, char** argv)
{
    // Variabiles to store the clock cicles used to mesure the execution time
    time_t start;
    time_t stop;
    double elapsed_time;

    // Set the parameters
    int image_size = 6;
    int num_channels = 4;
    int kernel_size = 3;
    int num_filters = 1;
    int stride = 2;

    // Read the image and store it in a tensor
    struct tensor img_tensor;
    img_tensor.row = image_size;
    img_tensor.col = image_size;
    img_tensor.depth = num_channels;
    img_tensor.data = (float*)malloc(img_tensor.row * img_tensor.col * img_tensor.depth * sizeof(float));
    init_tensor(&img_tensor);

    // Check img_tensor
    print_tensor(&img_tensor);

    // Print some information about the tensor
    printf("Image loaded successfully:\n");
    printf("Dimensions: %d x %d x %d\n", img_tensor.depth, img_tensor.row, img_tensor.col);

    // Declare the kernel
    struct tensor kernel;
    kernel.row = kernel_size;
    kernel.col = kernel_size;
    kernel.depth = num_channels;
    kernel.data = (float*)malloc(kernel.row * kernel.col * kernel.depth * sizeof(float));
    init_tensor(&kernel);

    // Check img_tensor
    print_tensor(&kernel);

    // GPU CONVOLUTION
    // Declare the structure to store the output of the convolution with GPU
    struct tensor output_tensor;
    output_tensor.col = img_tensor.col/stride;
    output_tensor.row = img_tensor.row/stride;
    output_tensor.depth = num_filters;
    output_tensor.data = (float*)malloc(output_tensor.row * output_tensor.col * output_tensor.depth * sizeof(float));

    // Fill the output data with zeros
    memset(output_tensor.data, 0, output_tensor.row * output_tensor.col * output_tensor.depth * sizeof(float));

    start = clock();
    // Perform convolution using GPU
    cudaError_t cudaStatus = Conv2dWithCuda(&img_tensor, &kernel, kernel_size, stride, num_channels, num_filters, &output_tensor);
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
    free_tensor(&kernel);
    free_tensor(&output_tensor);

    return 0;
}

