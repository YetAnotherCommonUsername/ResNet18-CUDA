#pragma once

#include <stdio.h>
#include <stdlib.h>
#include "tensor_utils.cuh"

void load_image_as_tensor(const char* filename, struct tensor* img_tensor);