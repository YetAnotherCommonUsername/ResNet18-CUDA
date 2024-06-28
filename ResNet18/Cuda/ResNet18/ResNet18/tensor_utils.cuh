#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

struct tensor {
	int col;
	int row;
	int depth;
	float* data;
};

void free_tensor(struct tensor* t);
void init_tensor(struct tensor* t);
void print_tensor(struct tensor* t);