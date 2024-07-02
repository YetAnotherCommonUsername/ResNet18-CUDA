#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "tensor_utils.cuh"

// void convolution_parallel(struct tensor* input_image, int* kernel, int kernel_size, struct tensor* output_image);

cudaError_t Conv2dWithCuda(struct tensor* input_tensor, struct tensor* kernels, int kernel_size, int stride, int num_filters, struct tensor* output_tensor);
__global__ void convolution_parallel(float* input_tensor, int nrow, int ncol, int nchannels, float* kernels, int kernel_size, int stride, float* output_tensor);

cudaError_t MaxPoolingWithCuda(struct tensor* input_tensor, int pool_size, int stride, struct tensor* output_tensor);
__global__ void max_pooling_parallel(float* input_tensor, int nrow, int ncol, int nchannels, int pool_size, int stride, float* output_tensor);

cudaError_t ResidualConnectionWithCuda(struct tensor* input_tensor1, struct tensor* input_tensor2, struct tensor* output_tensor);
__global__ void add_tensors_parallel(float* input_tensor1, float* input_tensor2, int nrow, int ncol, float* output_tensor);

cudaError_t FullyConnectedLayerWithCuda(float* input_array, float* weights, float* bias, int input_size, int num_classes, float* output_array);
__global__ void fully_connected_parallel(float* input_array, float* weights, float* bias, int input_size, int num_classes, float* output_array);

// cudaError_t ResNet(struct tensor* input_tensor, float* output_classes);