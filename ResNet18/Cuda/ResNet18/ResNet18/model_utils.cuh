#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "tensor_utils.cuh"

// void convolution_parallel(struct tensor* input_image, int* kernel, int kernel_size, struct tensor* output_image);

cudaError_t Conv2dWithCuda(struct tensor* input_tensor, struct tensor* kernels, int kernel_size, int stride, int num_channels, int num_filters, struct tensor* output_tensor);
__global__ void convolution_parallel(float* input_tensor, int nrow, int ncol, int nchannels, float* kernels, int kernel_size, int stride, float* output_tensor);

// cudaError_t ResNet(struct tensor* input_tensor, float* output_classes);