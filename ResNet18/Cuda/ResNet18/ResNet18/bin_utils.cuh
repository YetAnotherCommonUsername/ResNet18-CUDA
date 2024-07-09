#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tensor_utils.cuh"

void load_conv_weights(const char* filename, struct tensor* kernels, int kernel_size, int num_channels, int num_filters);
void load_matrix(const char* filename, float* weights, int ncol, int nrow);
void load_array(const char* filename, float* weights, int size);
char** load_classes(const char* filename, int num_classes);
void load_image_as_tensor(const char* filename, struct tensor* img_tensor);