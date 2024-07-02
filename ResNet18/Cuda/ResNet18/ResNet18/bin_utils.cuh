#pragma once

#include <stdio.h>
#include <stdlib.h>
#include "tensor_utils.cuh"

void load_conv_weights(const char* filename, struct tensor* kernels, int kernel_size, int num_channels, int num_filters);
void load_linear_weights(const char* filename, float* weights, int input_size, int num_classes);
void load_linear_bias(const char* filename, float* bias, int num_classes);