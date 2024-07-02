#pragma once

#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include "tensor_utils.cuh"

void load_image_as_tensor(const char* filename, struct tensor* img_tensor);
void load_weights(char* filename, struct tensor* kernel);