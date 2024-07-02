#pragma once

#include <stdio.h>
#include <stdlib.h>
#include "tensor_utils.cuh"

void load_weights(const char* filename, struct tensor* kernels, int kernel_size, int num_channels, int num_filters);