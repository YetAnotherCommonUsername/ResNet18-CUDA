#include "model_utils.cuh"

cudaError_t Conv2dWithCuda(struct tensor* input_tensor, struct tensor* kernels, int kernel_size, int stride, int num_filters, struct tensor* output_tensor) {
	// This function takes the struct input_image, the array kernel and perform the convolution using the GPU.
	// The resulting image will stored inside the struct 'output_image'.

	int nrow = input_tensor->row;
	int ncol = input_tensor->col;
	int nchannels = input_tensor->depth;

	int memsize_input_tensor = nchannels * nrow * ncol * sizeof(float);
	int memsize_kernels = num_filters * kernel_size * kernel_size * nchannels * sizeof(float);
	int memsize_output_tensor = output_tensor->col * output_tensor->row * output_tensor->depth * sizeof(float);

	// Declaration of the input_array, kernel and output_array and move them to the GPU.
	float* dev_input_data;
	cudaMalloc((void**)&dev_input_data, memsize_input_tensor);
	cudaMemcpy(dev_input_data, input_tensor->data, memsize_input_tensor, cudaMemcpyHostToDevice);

	float* dev_kernels;
	cudaMalloc((void**)&dev_kernels, memsize_kernels);
	for (int i = 0; i < num_filters; i++) {
		cudaMemcpy(dev_kernels + i * kernel_size * kernel_size * nchannels, kernels[i].data, kernel_size * kernel_size * nchannels * sizeof(float), cudaMemcpyHostToDevice);
	} 

	float* dev_output_data;
	cudaMalloc((void**)&dev_output_data, memsize_output_tensor);
	// No need to copy the output tensor to device before computation

	// Define CudaKernel settings
	dim3 threadInBlock(8, 8, 16); // Adjust to suitable block size
	dim3 numBlocks;
	numBlocks.x = (ncol + threadInBlock.x - 1) / threadInBlock.x;
	numBlocks.y = (nrow + threadInBlock.y - 1) / threadInBlock.y;
	numBlocks.z = num_filters;
	int memsize_shared_memory = threadInBlock.x * threadInBlock.y * threadInBlock.z * sizeof(float);

	// Get the starting time.
	cudaError_t cudaStatus;

	// Launch the cuda kernel that performs the convolution.
	convolution_parallel << <numBlocks, threadInBlock, memsize_shared_memory >> > (dev_input_data, nrow, ncol, nchannels, dev_kernels, kernel_size, stride, dev_output_data);

	// Compute the elapsed time in ms.
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "convolution_parallel() launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	// Move the output array from Device back to host.
	cudaMemcpy(output_tensor->data, dev_output_data, memsize_output_tensor, cudaMemcpyDeviceToHost);

	cudaFree(dev_input_data);
	cudaFree(dev_kernels);
	cudaFree(dev_output_data);

	return cudaStatus;
}

__global__ void convolution_parallel(float* input_tensor, int nrow, int ncol, int nchannels, float* kernels, int kernel_size, int stride, float* output_tensor) {
    // This function defines the kernel execute in every GPU's thread.
    // In the GPU version, we don't need the outer for loop to iterate over the all image. But, each thread operates on a single sub-image.

    extern __shared__ float shared_array[];

    // Compute the padding size
    int pad = kernel_size / 2;

    // Get the row, col, and channel of the image the thread is pointing on considering the stride factor.
    int row = (threadIdx.y + blockIdx.y * blockDim.y) * stride;
    int col = (threadIdx.x + blockIdx.x * blockDim.x) * stride;
    int tid = threadIdx.z;	
	int channel = tid;
	int shared_idx = tid * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	int kernel = blockIdx.z;

	// Initialization shared memory
	shared_array[shared_idx] = 0.0;

    // Ensure the pixel is inside a valid region of the image.
    if ((row < nrow) && (col < ncol) && (tid < nchannels)) {
        float result = 0.0f;

		// Convolution
		while (channel < nchannels) {
			for (int i = 0; i < kernel_size; i++) {
				for (int j = 0; j < kernel_size; j++) {
					int img_row = row + i - pad;
					int img_col = col + j - pad;

					// Handle padding by skipping invalid positions
					if (img_row >= 0 && img_row < nrow && img_col >= 0 && img_col < ncol) {
						int img_idx = channel * nrow * ncol + img_row * ncol + img_col;
						int kernel_idx = kernel * nchannels * kernel_size * kernel_size + channel * kernel_size * kernel_size + i * kernel_size + j;
						result += kernels[kernel_idx] * input_tensor[img_idx];
					}
				}
			}
			channel += blockDim.z;
		}
		shared_array[shared_idx] = result;
	}

    // Synchronize to ensure all threads have written to shared memory
    __syncthreads();

	// Reduction algorithm to sum results across all channels using the shared memory
	for (int s = blockDim.z / 2; s > 0; s >>= 1) {
		if (tid < s) {
			shared_array[shared_idx] += shared_array[shared_idx + s*(blockDim.x * blockDim.y)];
		}
		__syncthreads();
	}

	if ((row < nrow) && (col < ncol)) {
		if (tid == 0) {
			int output_nrow = (nrow + stride - 1) / stride;
			int output_ncol = (ncol + stride - 1) / stride;
			int output_image_idx = (kernel * output_nrow * output_ncol) + (blockIdx.y * blockDim.y + threadIdx.y) * (ncol / stride) + (blockIdx.x * blockDim.x + threadIdx.x);
			output_tensor[output_image_idx] = shared_array[threadIdx.y * blockDim.x + threadIdx.x];
		}
	}
}

void convolution_serial(struct tensor* input_tensor, struct tensor* kernels, int num_filters, int stride, struct tensor* output_tensor) {

	// Parametrs
	int kernel_size = kernels[0].row;
	int nrow = input_tensor->row;
	int ncol = input_tensor->col;
	int nchannels = input_tensor->depth;

	// Compute the padding size
	int pad = kernel_size / 2;

	// Initialize the output tensor to zero
	memset(output_tensor->data, 0, num_filters * (nrow / stride) * (ncol / stride) * sizeof(float));

	// Iterate over each kernel
	for (int k = 0; k < num_filters; k++) {
		// Iterate over the image with the stride
		for (int row = 0; row < nrow; row += stride) {
			for (int col = 0; col < ncol; col += stride) {
				float result = 0.0f;

				// Convolution for each channel
				for (int channel = 0; channel < nchannels; channel++) {
					for (int i = 0; i < kernel_size; i++) {
						for (int j = 0; j < kernel_size; j++) {
							int img_row = row + i - pad;
							int img_col = col + j - pad;

							// Handle padding by skipping invalid positions
							if (img_row >= 0 && img_row < nrow && img_col >= 0 && img_col < ncol) {
								int img_idx = channel * nrow * ncol + img_row * ncol + img_col;
								int kernel_idx =  channel * kernel_size * kernel_size + i * kernel_size + j;
								result += kernels[k].data[kernel_idx] * input_tensor->data[img_idx];
							}
						}
					}
				}

				// Calculate the output index and store the result
				int output_row = row / stride;
				int output_col = col / stride;
				int output_nrow = (nrow + stride - 1) / stride;
				int output_ncol = output_nrow;
				int output_idx = k * output_nrow * output_ncol + output_row * output_ncol + output_col;
				output_tensor->data[output_idx] = result;
			}
		}
	}
}

cudaError_t MaxPoolingWithCuda(struct tensor* input_tensor, int pool_size, int stride, struct tensor* output_tensor) {
	int nrow = input_tensor->row;
	int ncol = input_tensor->col;
	int nchannels = input_tensor->depth;

	int output_nrow = (nrow + stride - 1) / stride;
	int output_ncol = (ncol + stride - 1) / stride;

	int memsize_input_tensor = nrow * ncol * nchannels * sizeof(float);
	int memsize_output_tensor = output_nrow * output_ncol * nchannels * sizeof(float);

	float* dev_input_data;
	cudaMalloc((void**)&dev_input_data, memsize_input_tensor);
	cudaMemcpy(dev_input_data, input_tensor->data, memsize_input_tensor, cudaMemcpyHostToDevice);

	float* dev_output_data;
	cudaMalloc((void**)&dev_output_data, memsize_output_tensor);

	// Define CudaKernel settings.
	dim3 threadInBlock(16, 16);
	dim3 numBlocks;
	numBlocks.x = (ncol + threadInBlock.x - 1) / threadInBlock.x;
	numBlocks.y = (nrow + threadInBlock.y - 1) / threadInBlock.y;
	numBlocks.z = nchannels;

	max_pooling_parallel << <numBlocks, threadInBlock >> > (dev_input_data, nrow, ncol, nchannels, pool_size, stride, dev_output_data);

	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "max_pooling_parallel() launch failed : % s\n", cudaGetErrorString(cudaStatus));
	}

	cudaMemcpy(output_tensor->data, dev_output_data, memsize_output_tensor, cudaMemcpyDeviceToHost);

	cudaFree(dev_input_data);
	cudaFree(dev_output_data);

	return cudaStatus;
}

__global__ void max_pooling_parallel(float* input_tensor, int nrow, int ncol, int nchannels, int pool_size, int stride, float* output_tensor) {
	int row = (blockIdx.y * blockDim.y + threadIdx.y) * stride;
	int col = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
	int channel = blockIdx.z;

	// Compute the padding size
	int pad = pool_size / 2;

	if (row < nrow && col < ncol && channel < nchannels) {
		float max_value = -FLT_MAX;

		for (int i = 0; i < pool_size; i++) {
			for (int j = 0; j < pool_size; j++) {
				int img_row = row + i - pad;
				int img_col = col + j - pad;

				// Handle padding by skipping invalid positions
				if (img_row >= 0 && img_row < nrow && img_col >= 0 && img_col < ncol) {
					int img_idx = channel * nrow * ncol + img_row * ncol + img_col;
					max_value = fmaxf(max_value, input_tensor[img_idx]);
				}
			}
		}

		int output_nrow = (nrow + stride - 1) / stride;
		int output_ncol = (ncol + stride - 1) / stride;
		int output_idx = (channel * output_nrow * output_ncol) + (blockIdx.y * blockDim.y + threadIdx.y) * (ncol / stride) + (blockIdx.x * blockDim.x + threadIdx.x);
		output_tensor[output_idx] = max_value;
	}
}

void max_pooling_serial(struct tensor* input_tensor, int pool_size, int stride, struct tensor* output_tensor) {
	// Parametrs
	int nrow = input_tensor->row;
	int ncol = input_tensor->col;
	int nchannels = input_tensor->depth;

	// Compute the output dimensions
	int output_nrow = (nrow + stride - 1) / stride;
	int output_ncol = (ncol + stride - 1) / stride;

	// Compute the padding size
	int pad = pool_size / 2;

	// Iterate over each channel
	for (int channel = 0; channel < nchannels; channel++) {
		// Iterate over the image with the stride
		for (int row = 0; row < nrow; row += stride) {
			for (int col = 0; col < ncol; col += stride) {
				float max_value = -FLT_MAX;

				// Max pooling
				for (int i = 0; i < pool_size; i++) {
					for (int j = 0; j < pool_size; j++) {
						int img_row = row + i - pad;
						int img_col = col + j - pad;

						// Handle padding by skipping invalid positions
						if (img_row >= 0 && img_row < nrow && img_col >= 0 && img_col < ncol) {
							int img_idx = channel * nrow * ncol + img_row * ncol + img_col;
							if (input_tensor->data[img_idx] > max_value) {
								max_value = input_tensor->data[img_idx];
							}
						}
					}
				}

				// Calculate the output index and store the result
				int output_row = row / stride;
				int output_col = col / stride;
				int output_idx = channel * output_nrow * output_ncol + output_row * output_ncol + output_col;
				output_tensor->data[output_idx] = max_value;
			}
		}
	}
}

cudaError_t AveragePoolingWithCuda(struct tensor* input_tensor, float* output_array) {
	int nrow = input_tensor->row;
	int ncol = input_tensor->col;
	int nchannels = input_tensor->depth;

	int memsize_input_tensor = nrow * ncol * nchannels * sizeof(float);
	int memsize_output_tensor = nchannels * sizeof(float);

	float* dev_input_data;
	cudaMalloc((void**)&dev_input_data, memsize_input_tensor);
	cudaMemcpy(dev_input_data, input_tensor->data, memsize_input_tensor, cudaMemcpyHostToDevice);

	float* dev_output_data;
	cudaMalloc((void**)&dev_output_data, memsize_output_tensor);

	// Define CudaKernel settings.
	dim3 threadInBlock(16, 16);
	dim3 numBlocks;
	numBlocks.x = (ncol + threadInBlock.x - 1) / threadInBlock.x;
	numBlocks.y = (nrow + threadInBlock.y - 1) / threadInBlock.y;
	numBlocks.z = nchannels;
	int memsize_shared_memory = threadInBlock.x * threadInBlock.y * sizeof(float);

	average_pooling_parallel << <numBlocks, threadInBlock, memsize_shared_memory >> > (dev_input_data, nrow, ncol, nchannels, dev_output_data);

	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "average_pooling_parallel() launch failed : % s\n", cudaGetErrorString(cudaStatus));
	}

	cudaMemcpy(output_array, dev_output_data, memsize_output_tensor, cudaMemcpyDeviceToHost);

	cudaFree(dev_input_data);
	cudaFree(dev_output_data);

	return cudaStatus;
}

__global__ void average_pooling_parallel(float* input_tensor, int nrow, int ncol, int nchannels, float* output_tensor) {

	extern __shared__ float shared_array[];

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int channel = blockIdx.z;

	int tid = threadIdx.y * blockDim.x + threadIdx.x;
	
	// Initialization shared memory
	shared_array[tid] = 0.0;

	float result = 0.0f;

	while (row < nrow) {
		while (col < ncol) {
			int img_idx = channel * nrow * ncol + row * ncol + col;
			result += input_tensor[img_idx];
			col += blockDim.x;
		}
		row += blockDim.y;
	}
	shared_array[tid] = result;

	__syncthreads();

	// Compute reduction
	for (int s = (blockDim.x * blockDim.y) / 2; s > 0; s >>= 1) {
		if (tid < s) {
			shared_array[tid] += shared_array[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0) {
		int size = nrow * ncol;
		output_tensor[channel] = shared_array[0] / ((float) size);
	}
}

void average_pooling_serial(struct tensor* input_tensor, float* output_array) {
	// Parameters
	int nrow = input_tensor->row;
	int ncol = input_tensor->col;
	int nchannels = input_tensor->depth;

	// Iterate over each channel
	for (int channel = 0; channel < nchannels; channel++) {
		float sum = 0.0f;

		// Iterate over the image with the stride
		for (int row = 0; row < nrow; row++) {
			for (int col = 0; col < ncol; col++) {
				int img_idx = channel * nrow * ncol + row * ncol + col;
				sum += input_tensor->data[img_idx];
			}
		}

		// Calculate the output index and store the result
		float avg_value = sum / (float)(nrow * ncol);

		output_array[channel] = avg_value;
	}
}

cudaError_t ResidualConnectionWithCuda(struct tensor* input_tensor1, struct tensor* input_tensor2, struct tensor* output_tensor) {
	int nrow = input_tensor1->row;
	int ncol = input_tensor1->col;
	int nchannels = input_tensor1->depth;

	int memsize = nrow * ncol * nchannels * sizeof(float);

	float* dev_input_data1;
	cudaMalloc((void**)&dev_input_data1, memsize);
	cudaMemcpy(dev_input_data1, input_tensor1->data, memsize, cudaMemcpyHostToDevice);

	float* dev_input_data2;
	cudaMalloc((void**)&dev_input_data2, memsize);
	cudaMemcpy(dev_input_data2, input_tensor2->data, memsize, cudaMemcpyHostToDevice);

	float* dev_output_data;
	cudaMalloc((void**)&dev_output_data, memsize);

	// Define CudaKernel settings.
	dim3 threadInBlock(16, 16);
	dim3 numBlocks;
	numBlocks.x = (ncol + threadInBlock.x - 1) / threadInBlock.x;
	numBlocks.y = (nrow + threadInBlock.y - 1) / threadInBlock.y;
	numBlocks.z = nchannels;

	add_tensors_parallel << <numBlocks, threadInBlock >> > (dev_input_data1, dev_input_data2, nrow, ncol, dev_output_data);

	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "add_tensors_parallel() launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaMemcpy(output_tensor->data, dev_output_data, memsize, cudaMemcpyDeviceToHost);

	cudaFree(dev_input_data1);
	cudaFree(dev_input_data2);
	cudaFree(dev_output_data);

	return cudaStatus;
}

__global__ void add_tensors_parallel(float* input_tensor1, float* input_tensor2, int nrow, int ncol, float* output_tensor) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int channel = blockIdx.z;

	int tid = channel * nrow * ncol + row * ncol + col;

	if (row < nrow && col < ncol) {
		output_tensor[tid] = input_tensor1[tid] + input_tensor2[tid];
	}
}

void add_tensors_serial(struct tensor* input_tensor1, struct tensor* input_tensor2, struct tensor* output_tensor) {
	// Parameters
	int nrow = input_tensor1->row;
	int ncol = input_tensor1->col;
	int nchannels = input_tensor1->depth;

	int idx = 0;

	// Iterate over each the tensors
	for (int channel = 0; channel < nchannels; channel++) {
		for (int row = 0; row < nrow; row++) {
			for (int col = 0; col < ncol; col++) {
				idx = channel * nrow * ncol + row * ncol + col;
				output_tensor->data[idx] = input_tensor1->data[idx] + input_tensor2->data[idx];
			}
		}
	}
}

cudaError_t FullyConnectedLayerWithCuda(float* input_array, float* weights, float* bias, int input_size, int num_classes, float* output_array) {

	float* dev_input; 
	cudaMalloc((void**)&dev_input, input_size * sizeof(float));
	cudaMemcpy(dev_input, input_array, input_size * sizeof(float), cudaMemcpyHostToDevice);

	float* dev_weights;
	cudaMalloc((void**)&dev_weights, num_classes * input_size * sizeof(float));
	cudaMemcpy(dev_weights, weights, num_classes * input_size * sizeof(float), cudaMemcpyHostToDevice);

	float* dev_bias;
	cudaMalloc((void**)&dev_bias, num_classes * sizeof(float));
	cudaMemcpy(dev_bias, bias, num_classes * sizeof(float), cudaMemcpyHostToDevice);

	float* dev_output;
	cudaMalloc((void**)&dev_output, num_classes * sizeof(float));

	// Define CudaKernel settings.
	int threadInBlock = 32;		// Adjust to suitable block size
	int numBlocks = num_classes;
	int memsize_shared_memory = threadInBlock * sizeof(float);

	// Get the starting time.
	cudaError_t cudaStatus;

	// Launch the kernel
	fully_connected_parallel << <numBlocks, threadInBlock, memsize_shared_memory >> > (dev_input, dev_weights, dev_bias, input_size, num_classes, dev_output);

	// Compute the elapsed time in ms.
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "fully_connected_parallel() launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	// Copy the output data back to host
	cudaStatus = cudaMemcpy(output_array, dev_output, num_classes * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(dev_input);
	cudaFree(dev_weights);
	cudaFree(dev_bias);
	cudaFree(dev_output);

	return cudaStatus;
}

__global__ void fully_connected_parallel(float* input_array, float* weights, float* bias, int input_size, int num_classes, float* output_array) {

	extern __shared__ float shared_array[];

	int row = blockIdx.x;
	int tid = threadIdx.x;
	int col = tid;

	// Initialization shared memory
	shared_array[tid] = 0.0;

	float result = 0.0f;

	while (col < input_size) {
		result += input_array[col] * weights[row * input_size + col];
		col += blockDim.x;
	}
	shared_array[tid] = result;

	// Synchronize to ensure all threads have written to shared memory
	__syncthreads();

	// Reduction algorithm to sum results across all channels using the shared memory
	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			shared_array[tid] += shared_array[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0) {
		output_array[row] = shared_array[0] + bias[tid];
	}
}

void fully_connected_serial(float* input_tensor, float* weights, float* bias, int input_size, int num_classes, float* output_tensor) {
	// Perform matrix multiplication and add bias
	for (int i = 0; i < num_classes; i++) {
		float result = 0.0f;
		for (int j = 0; j < input_size; j++) {
			result += input_tensor[j] * weights[i * input_size + j];
		}
		output_tensor[i] = result + bias[i];
	}
}

void relu_serial(struct tensor* input_tensor, struct tensor* output_tensor) {
	// Parameters
	int nrow = input_tensor->row;
	int ncol = input_tensor->col;
	int nchannels = input_tensor->depth;

	memset(output_tensor->data, 0, nchannels * ncol * nrow * sizeof(float));

	int idx = 0;
	float val;

	// Iterate over each the tensor
	for (int channel = 0; channel < nchannels; channel++) {
		for (int row = 0; row < nrow; row++) {
			for (int col = 0; col < ncol; col++) {
				idx = channel * nrow * ncol + row * ncol + col;
				if ((val = input_tensor->data[idx]) > 0) {
					output_tensor->data[idx] = val;
				}
			}
		}
	}
}

cudaError_t ReLUWithCuda(struct tensor* input_tensor, struct tensor* output_tensor) {
	int nrow = input_tensor->row;
	int ncol = input_tensor->col;
	int nchannels = input_tensor->depth;

	int memsize = nrow * ncol * nchannels * sizeof(float);

	float* dev_input_data;
	cudaMalloc((void**)&dev_input_data, memsize);
	cudaMemcpy(dev_input_data, input_tensor->data, memsize, cudaMemcpyHostToDevice);

	float* dev_output_data;
	cudaMalloc((void**)&dev_output_data, memsize);

	// Define CudaKernel settings.
	dim3 threadInBlock(16, 16);
	dim3 numBlocks;
	numBlocks.x = (ncol + threadInBlock.x - 1) / threadInBlock.x;
	numBlocks.y = (nrow + threadInBlock.y - 1) / threadInBlock.y;
	numBlocks.z = nchannels;

	relu_parallel << <numBlocks, threadInBlock >> > (dev_input_data, nrow, ncol, dev_output_data);

	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "relu_parallel() launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaMemcpy(output_tensor->data, dev_output_data, memsize, cudaMemcpyDeviceToHost);

	cudaFree(dev_input_data);
	cudaFree(dev_output_data);

	return cudaStatus;
}

__global__ void relu_parallel(float* input_tensor, int nrow, int ncol, float* output_tensor) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int channel = blockIdx.z;

	int tid = channel * nrow * ncol + row * ncol + col;

	if (row < nrow && col < ncol) {
		output_tensor[tid] = fmaxf(0, input_tensor[tid]);
	}
}

cudaError_t BatchNormalizationWithCuda(struct tensor* input_tensor, float* beta, float* gamma, float* mean, float* std, struct tensor* output_tensor) {
	int nrow = input_tensor->row;
	int ncol = input_tensor->col;
	int nchannels = input_tensor->depth;

	int memsize = nrow * ncol * nchannels * sizeof(float);
	int memsize_parameters = nchannels * sizeof(float);

	float* dev_input_data;
	cudaMalloc((void**)&dev_input_data, memsize);
	cudaMemcpy(dev_input_data, input_tensor->data, memsize, cudaMemcpyHostToDevice);

	float* dev_beta;
	cudaMalloc((void**)&dev_beta, memsize_parameters);
	cudaMemcpy(dev_beta, beta, memsize_parameters, cudaMemcpyHostToDevice);

	float* dev_gamma;
	cudaMalloc((void**)&dev_gamma, memsize_parameters);
	cudaMemcpy(dev_gamma, gamma, memsize_parameters, cudaMemcpyHostToDevice);

	float* dev_mean;
	cudaMalloc((void**)&dev_mean, memsize_parameters);
	cudaMemcpy(dev_mean, mean, memsize_parameters, cudaMemcpyHostToDevice);

	float* dev_std;
	cudaMalloc((void**)&dev_std, memsize_parameters);
	cudaMemcpy(dev_std, std, memsize_parameters, cudaMemcpyHostToDevice);

	float* dev_output_data;
	cudaMalloc((void**)&dev_output_data, memsize);

	// Define CudaKernel settings.
	dim3 threadInBlock(16, 16);
	dim3 numBlocks;
	numBlocks.x = (ncol + threadInBlock.x - 1) / threadInBlock.x;
	numBlocks.y = (nrow + threadInBlock.y - 1) / threadInBlock.y;
	numBlocks.z = nchannels;

	batch_normalization_parallel << <numBlocks, threadInBlock >> > (dev_input_data, nrow, ncol, dev_beta, dev_gamma, dev_mean, dev_std, dev_output_data);

	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "batch_normalization_parallel() launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaMemcpy(output_tensor->data, dev_output_data, memsize, cudaMemcpyDeviceToHost);

	cudaFree(dev_input_data);
	cudaFree(dev_output_data);

	return cudaStatus;
}

__global__ void batch_normalization_parallel(float* input_tensor, int nrow, int ncol, float* beta, float* gamma, float* mean, float* std, float* output_tensor) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int channel = blockIdx.z;

	int tid = channel * nrow * ncol + row * ncol + col;

	if (row < nrow && col < ncol) {
		output_tensor[tid] = gamma[channel] * (input_tensor[tid] - mean[channel])/sqrtf(std[channel]) + beta[channel];
	}
}

void batch_normalization_serial(struct tensor* input_tensor, float* beta, float* gamma, float* mean, float* std, struct tensor* output_tensor) {
	// Parameters
	int nrow = input_tensor->row;
	int ncol = input_tensor->col;
	int nchannels = input_tensor->depth;

	int idx = 0;
	float val;

	// Iterate over each the tensor
	for (int channel = 0; channel < nchannels; channel++) {
		for (int row = 0; row < nrow; row++) {
			for (int col = 0; col < ncol; col++) {
				idx = channel * nrow * ncol + row * ncol + col;
				output_tensor->data[idx] = gamma[channel] * (input_tensor->data[idx] - mean[channel]) / sqrt(std[channel]) + beta[channel];
			}
		}
	}
}

void softmax_layer_serial(float* logits, int num_classes, float* probabilities) {

	float sum = 0.0f;

	for (int i = 0; i < num_classes; i++) {
		sum += exp(logits[i]);
	}

	for (int i = 0; i < num_classes; i++) {
		probabilities[i] = exp(logits[i]) / sum;
	}
}

__global__ void softmax_layer_parallel(float* logits, int num_classes, float* probabilities) {

	extern __shared__ float shared_array[];

	int tid = threadIdx.x;
	int col = tid;

	// Initialization shared memory
	shared_array[tid] = 0.0;

	float result = 0.0f;

	while (col < num_classes) {
		result += expf(logits[col]);
		col += blockDim.x;
	}
	shared_array[tid] = result;

	// Synchronize to ensure all threads have written to shared memory
	__syncthreads();

	// Reduction algorithm to sum results across all channels using the shared memory
	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			shared_array[tid] += shared_array[tid + s];
		}
		__syncthreads();
	}

	col = tid;

	while (col < num_classes) {
		probabilities[col] = expf(logits[col]) / shared_array[0];
		col += blockDim.x;
	}
	
}

void ResNet_serial(struct tensor* input_tensor, float* output_classes) {
	// Dimensions and sizes based on ResNet18 architecture
	int num_filters1 = 64, num_filters2 = 64, num_filters3 = 128, num_filters4 = 256, num_filters5 = 512;
	int kernel_size = 3, stride = 2, pool_size = 3, num_classes = 1000;

	// Debug
	bool debug = false;
	float* debug_data;
	int memsize_debug;

	// Allocate memory for weights and biases
	// conv1
	struct tensor* conv1_weights;
	float* conv1_batch1_beta, * conv1_batch1_gamma, * conv1_batch1_mean, * conv1_batch1_std;

	// conv2_x
	struct tensor* conv2_1_weights, * conv2_2_weights, * conv2_3_weights, * conv2_4_weights;
	float* conv2_batch1_beta, * conv2_batch1_gamma, * conv2_batch1_mean, * conv2_batch1_std;
	float* conv2_batch2_beta, * conv2_batch2_gamma, * conv2_batch2_mean, * conv2_batch2_std;
	float* conv2_batch3_beta, * conv2_batch3_gamma, * conv2_batch3_mean, * conv2_batch3_std;
	float* conv2_batch4_beta, * conv2_batch4_gamma, * conv2_batch4_mean, * conv2_batch4_std;

	// conv3_x
	struct tensor* conv3_1_weights, * conv3_2_weights, * conv3_3_weights, * conv3_4_weights, * conv3_5_weights;
	float* conv3_batch1_beta, * conv3_batch1_gamma, * conv3_batch1_mean, * conv3_batch1_std;
	float* conv3_batch2_beta, * conv3_batch2_gamma, * conv3_batch2_mean, * conv3_batch2_std;
	float* conv3_batch3_beta, * conv3_batch3_gamma, * conv3_batch3_mean, * conv3_batch3_std;
	float* conv3_batch4_beta, * conv3_batch4_gamma, * conv3_batch4_mean, * conv3_batch4_std;
	float* conv3_batch5_beta, * conv3_batch5_gamma, * conv3_batch5_mean, * conv3_batch5_std;

	// conv4_x
	struct tensor* conv4_1_weights, * conv4_2_weights, * conv4_3_weights, * conv4_4_weights, * conv4_5_weights;
	float* conv4_batch1_beta, * conv4_batch1_gamma, * conv4_batch1_mean, * conv4_batch1_std;
	float* conv4_batch2_beta, * conv4_batch2_gamma, * conv4_batch2_mean, * conv4_batch2_std;
	float* conv4_batch3_beta, * conv4_batch3_gamma, * conv4_batch3_mean, * conv4_batch3_std;
	float* conv4_batch4_beta, * conv4_batch4_gamma, * conv4_batch4_mean, * conv4_batch4_std;
	float* conv4_batch5_beta, * conv4_batch5_gamma, * conv4_batch5_mean, * conv4_batch5_std;

	// conv5_x
	struct tensor* conv5_1_weights, * conv5_2_weights, * conv5_3_weights, * conv5_4_weights, * conv5_5_weights;
	float* conv5_batch1_beta, * conv5_batch1_gamma, * conv5_batch1_mean, * conv5_batch1_std;
	float* conv5_batch2_beta, * conv5_batch2_gamma, * conv5_batch2_mean, * conv5_batch2_std;
	float* conv5_batch3_beta, * conv5_batch3_gamma, * conv5_batch3_mean, * conv5_batch3_std;
	float* conv5_batch4_beta, * conv5_batch4_gamma, * conv5_batch4_mean, * conv5_batch4_std;
	float* conv5_batch5_beta, * conv5_batch5_gamma, * conv5_batch5_mean, * conv5_batch5_std;

	// Fully connected
	float* fc_weights, * fc_bias;

	// Load weights from binary files
	// conv1
	int memsize_conv1_weights = num_filters1 * input_tensor->depth * 7 * 7 * sizeof(float);

	conv1_weights = (struct tensor*)malloc(memsize_conv1_weights + num_filters1 * 3 * sizeof(int));
	load_conv_weights("./../../../Parameters/conv_weights_0.bin", conv1_weights, 7, input_tensor->depth, num_filters1);

	conv1_batch1_beta = (float*)malloc(num_filters1 * sizeof(float));
	load_array("./../../../Parameters/batch_beta_1.bin", conv1_batch1_beta, num_filters1);
	conv1_batch1_gamma = (float*)malloc(num_filters1 * sizeof(float));
	load_array("./../../../Parameters/batch_gamma_1.bin", conv1_batch1_gamma, num_filters1);
	conv1_batch1_mean = (float*)malloc(num_filters1 * sizeof(float));
	load_array("./../../../Parameters/batch_mean_1.bin", conv1_batch1_mean, num_filters1);
	conv1_batch1_std = (float*)malloc(num_filters1 * sizeof(float));
	load_array("./../../../Parameters/batch_std_1.bin", conv1_batch1_std, num_filters1);

	// conv2_x
	int memsize_conv2_1_weights = num_filters2 * num_filters1 * kernel_size * kernel_size * sizeof(float);
	int memsize_conv2_2_weights = num_filters2 * num_filters2 * kernel_size * kernel_size * sizeof(float);
	int memsize_conv2_3_weights = num_filters2 * num_filters2 * kernel_size * kernel_size * sizeof(float);
	int memsize_conv2_4_weights = num_filters2 * num_filters2 * kernel_size * kernel_size * sizeof(float);

	conv2_1_weights = (struct tensor*)malloc(memsize_conv2_1_weights + num_filters2 * 3 * sizeof(int));
	load_conv_weights("./../../../Parameters/conv_weights_4.bin", conv2_1_weights, kernel_size, num_filters1, num_filters2);
	conv2_2_weights = (struct tensor*)malloc(memsize_conv2_2_weights + num_filters2 * 3 * sizeof(int));
	load_conv_weights("./../../../Parameters/conv_weights_7.bin", conv2_2_weights, kernel_size, num_filters2, num_filters2);
	conv2_3_weights = (struct tensor*)malloc(memsize_conv2_3_weights + num_filters2 * 3 * sizeof(int));
	load_conv_weights("./../../../Parameters/conv_weights_9.bin", conv2_3_weights, kernel_size, num_filters2, num_filters2);
	conv2_4_weights = (struct tensor*)malloc(memsize_conv2_4_weights + num_filters2 * 3 * sizeof(int));
	load_conv_weights("./../../../Parameters/conv_weights_12.bin", conv2_4_weights, kernel_size, num_filters2, num_filters2);

	conv2_batch1_beta = (float*)malloc(num_filters2 * sizeof(float));
	load_array("./../../../Parameters/batch_beta_5.bin", conv2_batch1_beta, num_filters2);
	conv2_batch1_gamma = (float*)malloc(num_filters2 * sizeof(float));
	load_array("./../../../Parameters/batch_gamma_5.bin", conv2_batch1_gamma, num_filters2);
	conv2_batch1_mean = (float*)malloc(num_filters2 * sizeof(float));
	load_array("./../../../Parameters/batch_mean_5.bin", conv2_batch1_mean, num_filters2);
	conv2_batch1_std = (float*)malloc(num_filters2 * sizeof(float));
	load_array("./../../../Parameters/batch_std_5.bin", conv2_batch1_std, num_filters2);

	conv2_batch2_beta = (float*)malloc(num_filters2 * sizeof(float));
	load_array("./../../../Parameters/batch_beta_8.bin", conv2_batch2_beta, num_filters2);
	conv2_batch2_gamma = (float*)malloc(num_filters2 * sizeof(float));
	load_array("./../../../Parameters/batch_gamma_8.bin", conv2_batch2_gamma, num_filters2);
	conv2_batch2_mean = (float*)malloc(num_filters2 * sizeof(float));
	load_array("./../../../Parameters/batch_mean_8.bin", conv2_batch2_mean, num_filters2);
	conv2_batch2_std = (float*)malloc(num_filters2 * sizeof(float));
	load_array("./../../../Parameters/batch_std_8.bin", conv2_batch2_std, num_filters2);

	conv2_batch3_beta = (float*)malloc(num_filters2 * sizeof(float));
	load_array("./../../../Parameters/batch_beta_10.bin", conv2_batch3_beta, num_filters2);
	conv2_batch3_gamma = (float*)malloc(num_filters2 * sizeof(float));
	load_array("./../../../Parameters/batch_gamma_10.bin", conv2_batch3_gamma, num_filters2);
	conv2_batch3_mean = (float*)malloc(num_filters2 * sizeof(float));
	load_array("./../../../Parameters/batch_mean_10.bin", conv2_batch3_mean, num_filters2);
	conv2_batch3_std = (float*)malloc(num_filters2 * sizeof(float));
	load_array("./../../../Parameters/batch_std_10.bin", conv2_batch3_std, num_filters2);

	conv2_batch4_beta = (float*)malloc(num_filters2 * sizeof(float));
	load_array("./../../../Parameters/batch_beta_13.bin", conv2_batch4_beta, num_filters2);
	conv2_batch4_gamma = (float*)malloc(num_filters2 * sizeof(float));
	load_array("./../../../Parameters/batch_gamma_13.bin", conv2_batch4_gamma, num_filters2);
	conv2_batch4_mean = (float*)malloc(num_filters2 * sizeof(float));
	load_array("./../../../Parameters/batch_mean_13.bin", conv2_batch4_mean, num_filters2);
	conv2_batch4_std = (float*)malloc(num_filters2 * sizeof(float));
	load_array("./../../../Parameters/batch_std_13.bin", conv2_batch4_std, num_filters2);

	// conv3_x
	int memsize_conv3_1_weights = num_filters3 * num_filters2 * kernel_size * kernel_size * sizeof(float);
	int memsize_conv3_2_weights = num_filters3 * num_filters3 * kernel_size * kernel_size * sizeof(float);
	int memsize_conv3_3_weights = num_filters3 * num_filters2 * 1 * 1 * sizeof(float);
	int memsize_conv3_4_weights = num_filters3 * num_filters3 * kernel_size * kernel_size * sizeof(float);
	int memsize_conv3_5_weights = num_filters3 * num_filters3 * kernel_size * kernel_size * sizeof(float);

	conv3_1_weights = (struct tensor*)malloc(memsize_conv3_1_weights + num_filters3 * 3 * sizeof(int));
	load_conv_weights("./../../../Parameters/conv_weights_14.bin", conv3_1_weights, kernel_size, num_filters2, num_filters3);
	conv3_2_weights = (struct tensor*)malloc(memsize_conv3_2_weights + num_filters3 * 3 * sizeof(int));
	load_conv_weights("./../../../Parameters/conv_weights_17.bin", conv3_2_weights, kernel_size, num_filters3, num_filters3);
	conv3_3_weights = (struct tensor*)malloc(memsize_conv3_3_weights + num_filters3 * 3 * sizeof(int));
	load_conv_weights("./../../../Parameters/conv_weights_19.bin", conv3_3_weights, 1, num_filters2, num_filters3);
	conv3_4_weights = (struct tensor*)malloc(memsize_conv3_4_weights + num_filters3 * 3 * sizeof(int));
	load_conv_weights("./../../../Parameters/conv_weights_21.bin", conv3_4_weights, kernel_size, num_filters3, num_filters3);
	conv3_5_weights = (struct tensor*)malloc(memsize_conv3_5_weights + num_filters3 * 3 * sizeof(int));
	load_conv_weights("./../../../Parameters/conv_weights_24.bin", conv3_5_weights, kernel_size, num_filters3, num_filters3);

	conv3_batch1_beta = (float*)malloc(num_filters3 * sizeof(float));
	load_array("./../../../Parameters/batch_beta_15.bin", conv3_batch1_beta, num_filters3);
	conv3_batch1_gamma = (float*)malloc(num_filters3 * sizeof(float));
	load_array("./../../../Parameters/batch_gamma_15.bin", conv3_batch1_gamma, num_filters3);
	conv3_batch1_mean = (float*)malloc(num_filters3 * sizeof(float));
	load_array("./../../../Parameters/batch_mean_15.bin", conv3_batch1_mean, num_filters3);
	conv3_batch1_std = (float*)malloc(num_filters3 * sizeof(float));
	load_array("./../../../Parameters/batch_std_15.bin", conv3_batch1_std, num_filters3);

	conv3_batch2_beta = (float*)malloc(num_filters3 * sizeof(float));
	load_array("./../../../Parameters/batch_beta_18.bin", conv3_batch2_beta, num_filters3);
	conv3_batch2_gamma = (float*)malloc(num_filters3 * sizeof(float));
	load_array("./../../../Parameters/batch_gamma_18.bin", conv3_batch2_gamma, num_filters3);
	conv3_batch2_mean = (float*)malloc(num_filters3 * sizeof(float));
	load_array("./../../../Parameters/batch_mean_18.bin", conv3_batch2_mean, num_filters3);
	conv3_batch2_std = (float*)malloc(num_filters3 * sizeof(float));
	load_array("./../../../Parameters/batch_std_18.bin", conv3_batch2_std, num_filters3);

	conv3_batch3_beta = (float*)malloc(num_filters3 * sizeof(float));
	load_array("./../../../Parameters/batch_beta_20.bin", conv3_batch3_beta, num_filters3);
	conv3_batch3_gamma = (float*)malloc(num_filters3 * sizeof(float));
	load_array("./../../../Parameters/batch_gamma_20.bin", conv3_batch3_gamma, num_filters3);
	conv3_batch3_mean = (float*)malloc(num_filters3 * sizeof(float));
	load_array("./../../../Parameters/batch_mean_20.bin", conv3_batch3_mean, num_filters3);
	conv3_batch3_std = (float*)malloc(num_filters3 * sizeof(float));
	load_array("./../../../Parameters/batch_std_20.bin", conv3_batch3_std, num_filters3);

	conv3_batch4_beta = (float*)malloc(num_filters3 * sizeof(float));
	load_array("./../../../Parameters/batch_beta_22.bin", conv3_batch4_beta, num_filters3);
	conv3_batch4_gamma = (float*)malloc(num_filters3 * sizeof(float));
	load_array("./../../../Parameters/batch_gamma_22.bin", conv3_batch4_gamma, num_filters3);
	conv3_batch4_mean = (float*)malloc(num_filters3 * sizeof(float));
	load_array("./../../../Parameters/batch_mean_22.bin", conv3_batch4_mean, num_filters3);
	conv3_batch4_std = (float*)malloc(num_filters3 * sizeof(float));
	load_array("./../../../Parameters/batch_std_22.bin", conv3_batch4_std, num_filters3);

	conv3_batch5_beta = (float*)malloc(num_filters3 * sizeof(float));
	load_array("./../../../Parameters/batch_beta_25.bin", conv3_batch5_beta, num_filters3);
	conv3_batch5_gamma = (float*)malloc(num_filters3 * sizeof(float));
	load_array("./../../../Parameters/batch_gamma_25.bin", conv3_batch5_gamma, num_filters3);
	conv3_batch5_mean = (float*)malloc(num_filters3 * sizeof(float));
	load_array("./../../../Parameters/batch_mean_25.bin", conv3_batch5_mean, num_filters3);
	conv3_batch5_std = (float*)malloc(num_filters3 * sizeof(float));
	load_array("./../../../Parameters/batch_std_25.bin", conv3_batch5_std, num_filters3);

	// conv4_x
	int memsize_conv4_1_weights = num_filters4 * num_filters3 * kernel_size * kernel_size * sizeof(float);
	int memsize_conv4_2_weights = num_filters4 * num_filters4 * kernel_size * kernel_size * sizeof(float);
	int memsize_conv4_3_weights = num_filters4 * num_filters3 * 1 * 1 * sizeof(float);
	int memsize_conv4_4_weights = num_filters4 * num_filters4 * kernel_size * kernel_size * sizeof(float);
	int memsize_conv4_5_weights = num_filters4 * num_filters4 * kernel_size * kernel_size * sizeof(float);

	conv4_1_weights = (struct tensor*)malloc(memsize_conv4_1_weights + num_filters4 * 3 * sizeof(int));
	load_conv_weights("./../../../Parameters/conv_weights_26.bin", conv4_1_weights, kernel_size, num_filters3, num_filters4);
	conv4_2_weights = (struct tensor*)malloc(memsize_conv4_2_weights + num_filters4 * 3 * sizeof(int));
	load_conv_weights("./../../../Parameters/conv_weights_29.bin", conv4_2_weights, kernel_size, num_filters4, num_filters4);
	conv4_3_weights = (struct tensor*)malloc(memsize_conv4_3_weights + num_filters4 * 3 * sizeof(int));
	load_conv_weights("./../../../Parameters/conv_weights_31.bin", conv4_3_weights, 1, num_filters3, num_filters4);
	conv4_4_weights = (struct tensor*)malloc(memsize_conv4_4_weights + num_filters4 * 3 * sizeof(int));
	load_conv_weights("./../../../Parameters/conv_weights_33.bin", conv4_4_weights, kernel_size, num_filters4, num_filters4);
	conv4_5_weights = (struct tensor*)malloc(memsize_conv4_5_weights + num_filters4 * 3 * sizeof(int));
	load_conv_weights("./../../../Parameters/conv_weights_36.bin", conv4_5_weights, kernel_size, num_filters4, num_filters4);

	conv4_batch1_beta = (float*)malloc(num_filters4 * sizeof(float));
	load_array("./../../../Parameters/batch_beta_27.bin", conv4_batch1_beta, num_filters4);
	conv4_batch1_gamma = (float*)malloc(num_filters4 * sizeof(float));
	load_array("./../../../Parameters/batch_gamma_27.bin", conv4_batch1_gamma, num_filters4);
	conv4_batch1_mean = (float*)malloc(num_filters4 * sizeof(float));
	load_array("./../../../Parameters/batch_mean_27.bin", conv4_batch1_mean, num_filters4);
	conv4_batch1_std = (float*)malloc(num_filters4 * sizeof(float));
	load_array("./../../../Parameters/batch_std_27.bin", conv4_batch1_std, num_filters4);

	conv4_batch2_beta = (float*)malloc(num_filters4 * sizeof(float));
	load_array("./../../../Parameters/batch_beta_30.bin", conv4_batch2_beta, num_filters4);
	conv4_batch2_gamma = (float*)malloc(num_filters4 * sizeof(float));
	load_array("./../../../Parameters/batch_gamma_30.bin", conv4_batch2_gamma, num_filters4);
	conv4_batch2_mean = (float*)malloc(num_filters4 * sizeof(float));
	load_array("./../../../Parameters/batch_mean_30.bin", conv4_batch2_mean, num_filters4);
	conv4_batch2_std = (float*)malloc(num_filters4 * sizeof(float));
	load_array("./../../../Parameters/batch_std_30.bin", conv4_batch2_std, num_filters4);

	conv4_batch3_beta = (float*)malloc(num_filters4 * sizeof(float));
	load_array("./../../../Parameters/batch_beta_32.bin", conv4_batch3_beta, num_filters4);
	conv4_batch3_gamma = (float*)malloc(num_filters4 * sizeof(float));
	load_array("./../../../Parameters/batch_gamma_32.bin", conv4_batch3_gamma, num_filters4);
	conv4_batch3_mean = (float*)malloc(num_filters4 * sizeof(float));
	load_array("./../../../Parameters/batch_mean_32.bin", conv4_batch3_mean, num_filters4);
	conv4_batch3_std = (float*)malloc(num_filters4 * sizeof(float));
	load_array("./../../../Parameters/batch_std_32.bin", conv4_batch3_std, num_filters4);

	conv4_batch4_beta = (float*)malloc(num_filters4 * sizeof(float));
	load_array("./../../../Parameters/batch_beta_34.bin", conv4_batch4_beta, num_filters4);
	conv4_batch4_gamma = (float*)malloc(num_filters4 * sizeof(float));
	load_array("./../../../Parameters/batch_gamma_34.bin", conv4_batch4_gamma, num_filters4);
	conv4_batch4_mean = (float*)malloc(num_filters4 * sizeof(float));
	load_array("./../../../Parameters/batch_mean_34.bin", conv4_batch4_mean, num_filters4);
	conv4_batch4_std = (float*)malloc(num_filters4 * sizeof(float));
	load_array("./../../../Parameters/batch_std_34.bin", conv4_batch4_std, num_filters4);

	conv4_batch5_beta = (float*)malloc(num_filters4 * sizeof(float));
	load_array("./../../../Parameters/batch_beta_37.bin", conv4_batch5_beta, num_filters4);
	conv4_batch5_gamma = (float*)malloc(num_filters4 * sizeof(float));
	load_array("./../../../Parameters/batch_gamma_37.bin", conv4_batch5_gamma, num_filters4);
	conv4_batch5_mean = (float*)malloc(num_filters4 * sizeof(float));
	load_array("./../../../Parameters/batch_mean_37.bin", conv4_batch5_mean, num_filters4);
	conv4_batch5_std = (float*)malloc(num_filters4 * sizeof(float));
	load_array("./../../../Parameters/batch_std_37.bin", conv4_batch5_std, num_filters4);

	// conv5_x
	int memsize_conv5_1_weights = num_filters5 * num_filters4 * kernel_size * kernel_size * sizeof(float);
	int memsize_conv5_2_weights = num_filters5 * num_filters5 * kernel_size * kernel_size * sizeof(float);
	int memsize_conv5_3_weights = num_filters5 * num_filters4 * 1 * 1 * sizeof(float);
	int memsize_conv5_4_weights = num_filters5 * num_filters5 * kernel_size * kernel_size * sizeof(float);
	int memsize_conv5_5_weights = num_filters5 * num_filters5 * kernel_size * kernel_size * sizeof(float);

	conv5_1_weights = (struct tensor*)malloc(memsize_conv5_1_weights + num_filters5 * 3 * sizeof(int));
	load_conv_weights("./../../../Parameters/conv_weights_38.bin", conv5_1_weights, kernel_size, num_filters4, num_filters5);
	conv5_2_weights = (struct tensor*)malloc(memsize_conv5_2_weights + num_filters5 * 3 * sizeof(int));
	load_conv_weights("./../../../Parameters/conv_weights_41.bin", conv5_2_weights, kernel_size, num_filters5, num_filters5);
	conv5_3_weights = (struct tensor*)malloc(memsize_conv5_3_weights + num_filters5 * 3 * sizeof(int));
	load_conv_weights("./../../../Parameters/conv_weights_43.bin", conv5_3_weights, 1, num_filters4, num_filters5);
	conv5_4_weights = (struct tensor*)malloc(memsize_conv5_4_weights + num_filters5 * 3 * sizeof(int));
	load_conv_weights("./../../../Parameters/conv_weights_45.bin", conv5_4_weights, kernel_size, num_filters5, num_filters5);
	conv5_5_weights = (struct tensor*)malloc(memsize_conv5_5_weights + num_filters5 * 3 * sizeof(int));
	load_conv_weights("./../../../Parameters/conv_weights_48.bin", conv5_5_weights, kernel_size, num_filters5, num_filters5);

	conv5_batch1_beta = (float*)malloc(num_filters5 * sizeof(float));
	load_array("./../../../Parameters/batch_beta_39.bin", conv5_batch1_beta, num_filters5);
	conv5_batch1_gamma = (float*)malloc(num_filters5 * sizeof(float));
	load_array("./../../../Parameters/batch_gamma_39.bin", conv5_batch1_gamma, num_filters5);
	conv5_batch1_mean = (float*)malloc(num_filters5 * sizeof(float));
	load_array("./../../../Parameters/batch_mean_39.bin", conv5_batch1_mean, num_filters5);
	conv5_batch1_std = (float*)malloc(num_filters5 * sizeof(float));
	load_array("./../../../Parameters/batch_std_39.bin", conv5_batch1_std, num_filters5);

	conv5_batch2_beta = (float*)malloc(num_filters5 * sizeof(float));
	load_array("./../../../Parameters/batch_beta_42.bin", conv5_batch2_beta, num_filters5);
	conv5_batch2_gamma = (float*)malloc(num_filters5 * sizeof(float));
	load_array("./../../../Parameters/batch_gamma_42.bin", conv5_batch2_gamma, num_filters5);
	conv5_batch2_mean = (float*)malloc(num_filters5 * sizeof(float));
	load_array("./../../../Parameters/batch_mean_42.bin", conv5_batch2_mean, num_filters5);
	conv5_batch2_std = (float*)malloc(num_filters5 * sizeof(float));
	load_array("./../../../Parameters/batch_std_42.bin", conv5_batch2_std, num_filters5);

	conv5_batch3_beta = (float*)malloc(num_filters5 * sizeof(float));
	load_array("./../../../Parameters/batch_beta_44.bin", conv5_batch3_beta, num_filters5);
	conv5_batch3_gamma = (float*)malloc(num_filters5 * sizeof(float));
	load_array("./../../../Parameters/batch_gamma_44.bin", conv5_batch3_gamma, num_filters5);
	conv5_batch3_mean = (float*)malloc(num_filters5 * sizeof(float));
	load_array("./../../../Parameters/batch_mean_44.bin", conv5_batch3_mean, num_filters5);
	conv5_batch3_std = (float*)malloc(num_filters5 * sizeof(float));
	load_array("./../../../Parameters/batch_std_44.bin", conv5_batch3_std, num_filters5);

	conv5_batch4_beta = (float*)malloc(num_filters5 * sizeof(float));
	load_array("./../../../Parameters/batch_beta_46.bin", conv5_batch4_beta, num_filters5);
	conv5_batch4_gamma = (float*)malloc(num_filters5 * sizeof(float));
	load_array("./../../../Parameters/batch_gamma_46.bin", conv5_batch4_gamma, num_filters5);
	conv5_batch4_mean = (float*)malloc(num_filters5 * sizeof(float));
	load_array("./../../../Parameters/batch_mean_46.bin", conv5_batch4_mean, num_filters5);
	conv5_batch4_std = (float*)malloc(num_filters5 * sizeof(float));
	load_array("./../../../Parameters/batch_std_46.bin", conv5_batch4_std, num_filters5);

	conv5_batch5_beta = (float*)malloc(num_filters5 * sizeof(float));
	load_array("./../../../Parameters/batch_beta_49.bin", conv5_batch5_beta, num_filters5);
	conv5_batch5_gamma = (float*)malloc(num_filters5 * sizeof(float));
	load_array("./../../../Parameters/batch_gamma_49.bin", conv5_batch5_gamma, num_filters5);
	conv5_batch5_mean = (float*)malloc(num_filters5 * sizeof(float));
	load_array("./../../../Parameters/batch_mean_49.bin", conv5_batch5_mean, num_filters5);
	conv5_batch5_std = (float*)malloc(num_filters5 * sizeof(float));
	load_array("./../../../Parameters/batch_std_49.bin", conv5_batch5_std, num_filters5);

	// Fully connected
	fc_weights = (float*)malloc(num_filters5 * num_classes * sizeof(float));
	load_matrix("./../../../Parameters/linear_weights_51.bin", fc_weights, num_filters5, num_classes);
	fc_bias = (float*)malloc(num_classes * sizeof(float));
	load_array("./../../../Parameters/linear_bias_51.bin", fc_bias, num_classes);

	// Network architecture
	//=================================
	//							 Layer 1 
	//=================================
	// Define the hyperparameters
	int image_size = input_tensor->col;
	int num_channel = input_tensor->depth;

	// Device memory for intermediate tensors
	image_size = (image_size + stride - 1) / stride;
	int memsize_output_data = num_filters1 * image_size * image_size * sizeof(float);

	struct tensor layer1_output_data1;
	layer1_output_data1.row = image_size;
	layer1_output_data1.col = image_size;
	layer1_output_data1.depth = num_filters1;
	layer1_output_data1.data = (float*)malloc(memsize_output_data);

	struct tensor layer1_output_data2;
	layer1_output_data2.row = image_size;
	layer1_output_data2.col = image_size;
	layer1_output_data2.depth = num_filters1;
	layer1_output_data2.data = (float*)malloc(memsize_output_data);

	struct tensor layer1_output_data3;
	layer1_output_data3.row = image_size;
	layer1_output_data3.col = image_size;
	layer1_output_data3.depth = num_filters1;
	layer1_output_data3.data = (float*)malloc(memsize_output_data);

	// conv1
	convolution_serial(input_tensor, conv1_weights, num_filters1, stride, &layer1_output_data1);

	// BatchNorm
	batch_normalization_serial(&layer1_output_data1, conv1_batch1_beta, conv1_batch1_gamma, conv1_batch1_mean, conv1_batch1_std, &layer1_output_data2);

	// ReLU
	relu_serial(&layer1_output_data2, &layer1_output_data1);

	//=================================
	//							 Layer 2 
	//=================================
	// Device memory for intermediate tensors
	image_size = (image_size + stride - 1) / stride;
	memsize_output_data = num_filters2 * image_size * image_size * sizeof(float);

	struct tensor layer2_output_data1;
	layer2_output_data1.row = image_size;
	layer2_output_data1.col = image_size;
	layer2_output_data1.depth = num_filters2;
	layer2_output_data1.data = (float*)malloc(memsize_output_data);

	struct tensor layer2_output_data2;
	layer2_output_data2.row = image_size;
	layer2_output_data2.col = image_size;
	layer2_output_data2.depth = num_filters2;
	layer2_output_data2.data = (float*)malloc(memsize_output_data);

	struct tensor layer2_output_data3;
	layer2_output_data3.row = image_size;
	layer2_output_data3.col = image_size;
	layer2_output_data3.depth = num_filters2;
	layer2_output_data3.data = (float*)malloc(memsize_output_data);

	// MaxPool
	max_pooling_serial(&layer1_output_data1, pool_size, stride, &layer2_output_data1);

	// ReLU (residual connection)
	relu_serial(&layer2_output_data1, &layer2_output_data3);

	// conv2_1
	convolution_serial(&layer2_output_data1, conv2_1_weights, num_filters2, 1, &layer2_output_data2);

	// BatchNorm
	batch_normalization_serial(&layer2_output_data2, conv2_batch1_beta, conv2_batch1_gamma, conv2_batch1_mean, conv2_batch1_std, &layer2_output_data1);

	// ReLU
	relu_serial(&layer2_output_data1, &layer2_output_data2);

	// conv2_2
	convolution_serial(&layer2_output_data2, conv2_2_weights, num_filters2, 1, &layer2_output_data1);

	// BatchNorm
	batch_normalization_serial(&layer2_output_data1, conv2_batch2_beta, conv2_batch2_gamma, conv2_batch2_mean, conv2_batch2_std, &layer2_output_data2);

	// ResidualConnection
	add_tensors_serial(&layer2_output_data2, &layer2_output_data3, &layer2_output_data1);

	// ReLU
	relu_serial(&layer2_output_data1, &layer2_output_data2);

	// Identity block 2
	// ReLU (residual connection)
	relu_serial(&layer2_output_data2, &layer2_output_data3);

	// conv2_3
	convolution_serial(&layer2_output_data2, conv2_3_weights, num_filters2, 1, &layer2_output_data1);

	// BatchNorm
	batch_normalization_serial(&layer2_output_data1, conv2_batch3_beta, conv2_batch3_gamma, conv2_batch3_mean, conv2_batch3_std, &layer2_output_data2);

	// ReLU
	relu_serial(&layer2_output_data2, &layer2_output_data1);

	// conv2_4
	convolution_serial(&layer2_output_data1, conv2_4_weights, num_filters2, 1, &layer2_output_data2);

	// BatchNorm
	batch_normalization_serial(&layer2_output_data2, conv2_batch4_beta, conv2_batch4_gamma, conv2_batch4_mean, conv2_batch4_std, &layer2_output_data1);

	// ResidualConnection
	add_tensors_serial(&layer2_output_data1, &layer2_output_data3, &layer2_output_data2);

	// ReLU
	relu_serial(&layer2_output_data2, &layer2_output_data1);

	//=================================
	//							 Layer 3 
	//=================================
	// Device memory for intermediate tensors
	image_size = (image_size + stride - 1) / stride;
	memsize_output_data = num_filters3 * image_size * image_size * sizeof(float);

	struct tensor layer3_output_data1;
	layer3_output_data1.row = image_size;
	layer3_output_data1.col = image_size;
	layer3_output_data1.depth = num_filters3;
	layer3_output_data1.data = (float*)malloc(memsize_output_data);

	struct tensor layer3_output_data2;
	layer3_output_data2.row = image_size;
	layer3_output_data2.col = image_size;
	layer3_output_data2.depth = num_filters3;
	layer3_output_data2.data = (float*)malloc(memsize_output_data);

	struct tensor layer3_output_data3;
	layer3_output_data3.row = image_size;
	layer3_output_data3.col = image_size;
	layer3_output_data3.depth = num_filters3;
	layer3_output_data3.data = (float*)malloc(memsize_output_data);

	struct tensor layer3_output_data4;
	layer3_output_data4.row = image_size;
	layer3_output_data4.col = image_size;
	layer3_output_data4.depth = num_filters3;
	layer3_output_data4.data = (float*)malloc(memsize_output_data);

	// Convolution block
	// conv3_1
	convolution_serial(&layer2_output_data1, conv3_1_weights, num_filters3, stride, &layer3_output_data1);

	// conv3_3 (residual connection)
	convolution_serial(&layer2_output_data1, conv3_3_weights, num_filters3, stride, &layer3_output_data3);

	// BatchNorm
	batch_normalization_serial(&layer3_output_data1, conv3_batch1_beta, conv3_batch1_gamma, conv3_batch1_mean, conv3_batch1_std, &layer3_output_data2);

	// ReLU
	relu_serial(&layer3_output_data2, &layer3_output_data1);

	// BatchNorm (residual_connection)
	batch_normalization_serial(&layer3_output_data3, conv3_batch3_beta, conv3_batch3_gamma, conv3_batch3_mean, conv3_batch3_std, &layer3_output_data4);

	// conv3_2
	convolution_serial(&layer3_output_data1, conv3_2_weights, num_filters3, 1, &layer3_output_data2);

	// BatchNorm
	batch_normalization_serial(&layer3_output_data2, conv3_batch2_beta, conv3_batch2_gamma, conv3_batch2_mean, conv3_batch2_std, &layer3_output_data1);

	// ResidualConnection
	add_tensors_serial(&layer3_output_data1, &layer3_output_data4, &layer3_output_data2);

	// ReLU
	relu_serial(&layer3_output_data2, &layer3_output_data1);

	// Identity block
	// ReLU (residual connection)
	relu_serial(&layer3_output_data1, &layer3_output_data3);

	// conv3_4
	convolution_serial(&layer3_output_data1, conv3_4_weights, num_filters3, 1, &layer3_output_data2);

	// BatchNorm
	batch_normalization_serial(&layer3_output_data2, conv3_batch4_beta, conv3_batch4_gamma, conv3_batch4_mean, conv3_batch4_std, &layer3_output_data1);

	// ReLU
	relu_serial(&layer3_output_data1, &layer3_output_data2);

	// conv3_5
	convolution_serial(&layer3_output_data2, conv3_5_weights, num_filters3, 1, &layer3_output_data1);

	// BatchNorm
	batch_normalization_serial(&layer3_output_data1, conv3_batch5_beta, conv3_batch5_gamma, conv3_batch5_mean, conv3_batch5_std, &layer3_output_data2);

	// ResidualConnection
	add_tensors_serial(&layer3_output_data2, &layer3_output_data3, &layer3_output_data1);

	// ReLU
	relu_serial(&layer3_output_data1, &layer3_output_data2);

	//=================================
	//							 Layer 4 
	//=================================
	// Device memory for intermediate tensors
	image_size = (image_size + stride - 1) / stride;
	memsize_output_data = num_filters4 * image_size * image_size * sizeof(float);

	struct tensor layer4_output_data1;
	layer4_output_data1.row = image_size;
	layer4_output_data1.col = image_size;
	layer4_output_data1.depth = num_filters4;
	layer4_output_data1.data = (float*)malloc(memsize_output_data);

	struct tensor layer4_output_data2;
	layer4_output_data2.row = image_size;
	layer4_output_data2.col = image_size;
	layer4_output_data2.depth = num_filters4;
	layer4_output_data2.data = (float*)malloc(memsize_output_data);

	struct tensor layer4_output_data3;
	layer4_output_data3.row = image_size;
	layer4_output_data3.col = image_size;
	layer4_output_data3.depth = num_filters4;
	layer4_output_data3.data = (float*)malloc(memsize_output_data);

	struct tensor layer4_output_data4;
	layer4_output_data4.row = image_size;
	layer4_output_data4.col = image_size;
	layer4_output_data4.depth = num_filters4;
	layer4_output_data4.data = (float*)malloc(memsize_output_data);

	// Convolution block
	// conv4_1
	convolution_serial(&layer3_output_data2, conv4_1_weights, num_filters4, stride, &layer4_output_data1);

	// conv4_3 (residual connection)
	convolution_serial(&layer3_output_data2, conv4_3_weights, num_filters4, stride, &layer4_output_data3);

	// BatchNorm
	batch_normalization_serial(&layer4_output_data1, conv4_batch1_beta, conv4_batch1_gamma, conv4_batch1_mean, conv4_batch1_std, &layer4_output_data2);

	// ReLU
	relu_serial(&layer4_output_data2, &layer4_output_data1);

	// BatchNorm (residual_connection)
	batch_normalization_serial(&layer4_output_data3, conv4_batch3_beta, conv4_batch3_gamma, conv4_batch3_mean, conv4_batch3_std, &layer4_output_data4);

	// conv4_2
	convolution_serial(&layer4_output_data1, conv4_2_weights, num_filters4, 1, &layer4_output_data2);

	// BatchNorm
	batch_normalization_serial(&layer4_output_data2, conv4_batch2_beta, conv4_batch2_gamma, conv4_batch2_mean, conv4_batch2_std, &layer4_output_data1);

	// ResidualConnection
	add_tensors_serial(&layer4_output_data1, &layer4_output_data4, &layer4_output_data2);

	// ReLU
	relu_serial(&layer4_output_data2, &layer4_output_data1);

	// Identity block
	// ReLU (residual connection)
	relu_serial(&layer4_output_data1, &layer4_output_data3);

	// conv4_4
	convolution_serial(&layer4_output_data1, conv4_4_weights, num_filters4, 1, &layer4_output_data2);

	// BatchNorm
	batch_normalization_serial(&layer4_output_data2, conv4_batch4_beta, conv4_batch4_gamma, conv4_batch4_mean, conv4_batch4_std, &layer4_output_data1);

	// ReLU
	relu_serial(&layer4_output_data1, &layer4_output_data2);

	// conv4_5
	convolution_serial(&layer4_output_data2, conv4_5_weights, num_filters4, 1, &layer4_output_data1);

	// BatchNorm
	batch_normalization_serial(&layer4_output_data1, conv4_batch5_beta, conv4_batch5_gamma, conv4_batch5_mean, conv4_batch5_std, &layer4_output_data2);

	// ResidualConnection
	add_tensors_serial(&layer4_output_data2, &layer4_output_data3, &layer4_output_data1);

	// ReLU
	relu_serial(&layer4_output_data1, &layer4_output_data2);

	//=================================
	//							 Layer 5 
	//=================================
	//  Device memory for intermediate tensors
	image_size = (image_size + stride - 1) / stride;
	memsize_output_data = num_filters5 * image_size * image_size * sizeof(float);

	struct tensor layer5_output_data1;
	layer5_output_data1.row = image_size;
	layer5_output_data1.col = image_size;
	layer5_output_data1.depth = num_filters5;
	layer5_output_data1.data = (float*)malloc(memsize_output_data);

	struct tensor layer5_output_data2;
	layer5_output_data2.row = image_size;
	layer5_output_data2.col = image_size;
	layer5_output_data2.depth = num_filters5;
	layer5_output_data2.data = (float*)malloc(memsize_output_data);

	struct tensor layer5_output_data3;
	layer5_output_data3.row = image_size;
	layer5_output_data3.col = image_size;
	layer5_output_data3.depth = num_filters5;
	layer5_output_data3.data = (float*)malloc(memsize_output_data);

	struct tensor layer5_output_data4;
	layer5_output_data4.row = image_size;
	layer5_output_data4.col = image_size;
	layer5_output_data4.depth = num_filters5;
	layer5_output_data4.data = (float*)malloc(memsize_output_data);

	// Convolution block
	// conv5_1
	convolution_serial(&layer4_output_data2, conv5_1_weights, num_filters5, stride, &layer5_output_data1);

	// conv5_3 (residual connection)
	convolution_serial(&layer4_output_data2, conv5_3_weights, num_filters5, stride, &layer5_output_data3);

	// BatchNorm
	batch_normalization_serial(&layer5_output_data1, conv5_batch1_beta, conv5_batch1_gamma, conv5_batch1_mean, conv5_batch1_std, &layer5_output_data2);

	// ReLU
	relu_serial(&layer5_output_data2, &layer5_output_data1);

	// BatchNorm (residual_connection)
	batch_normalization_serial(&layer5_output_data3, conv5_batch3_beta, conv5_batch3_gamma, conv5_batch3_mean, conv5_batch3_std, &layer5_output_data4);

	// conv5_2
	convolution_serial(&layer5_output_data1, conv5_2_weights, num_filters5, 1, &layer5_output_data2);

	// BatchNorm
	batch_normalization_serial(&layer5_output_data2, conv5_batch2_beta, conv5_batch2_gamma, conv5_batch2_mean, conv5_batch2_std, &layer5_output_data1);

	// ResidualConnection
	add_tensors_serial(&layer5_output_data1, &layer5_output_data4, &layer5_output_data2);

	// ReLU
	relu_serial(&layer5_output_data2, &layer5_output_data1);

	// Identity block
	// ReLU (residual connection)
	relu_serial(&layer5_output_data1, &layer5_output_data3);

	// conv5_4
	convolution_serial(&layer5_output_data1, conv5_4_weights, num_filters5, 1, &layer5_output_data2);

	// BatchNorm
	batch_normalization_serial(&layer5_output_data2, conv5_batch4_beta, conv5_batch4_gamma, conv5_batch4_mean, conv5_batch4_std, &layer5_output_data1);

	// ReLU
	relu_serial(&layer5_output_data1, &layer5_output_data2);

	// conv5_5
	convolution_serial(&layer5_output_data2, conv5_5_weights, num_filters5, 1, &layer5_output_data1);

	// BatchNorm
	batch_normalization_serial(&layer5_output_data1, conv5_batch5_beta, conv5_batch5_gamma, conv5_batch5_mean, conv5_batch5_std, &layer5_output_data2);

	// ResidualConnection
	add_tensors_serial(&layer5_output_data2, &layer5_output_data3, &layer5_output_data1);

	// ReLU
	relu_serial(&layer5_output_data1, &layer5_output_data2);

	// Final average pooling and fully connected layer
	// Device memory for intermediate tensors
	float* flatten_data;
	flatten_data = (float*)malloc(num_filters5 * sizeof(float));

	float* logits_data;
	logits_data = (float*)malloc(num_classes * sizeof(float));

	// AveragePool
	average_pooling_serial(&layer5_output_data2, flatten_data);

	// Fully connected layer
	fully_connected_serial(flatten_data, fc_weights, fc_bias, num_filters5, num_classes, logits_data);

	// Softmax
	softmax_layer_serial(logits_data, num_classes, output_classes);

	// Free allocated memory (host)
	// conv1
	for (int i = 0; i < num_filters1; i++) {
		free_tensor(&conv1_weights[i]);
	}

	free(conv1_batch1_beta);
	free(conv1_batch1_gamma);
	free(conv1_batch1_mean);
	free(conv1_batch1_std);

	// conv2_x
	for (int i = 0; i < num_filters2; i++) {
		free_tensor(&conv2_1_weights[i]);
		free_tensor(&conv2_2_weights[i]);
		free_tensor(&conv2_3_weights[i]);
		free_tensor(&conv2_4_weights[i]);
	}

	free(conv2_batch1_beta);
	free(conv2_batch1_gamma);
	free(conv2_batch1_mean);
	free(conv2_batch1_std);

	free(conv2_batch2_beta);
	free(conv2_batch2_gamma);
	free(conv2_batch2_mean);
	free(conv2_batch2_std);

	free(conv2_batch3_beta);
	free(conv2_batch3_gamma);
	free(conv2_batch3_mean);
	free(conv2_batch3_std);

	free(conv2_batch4_beta);
	free(conv2_batch4_gamma);
	free(conv2_batch4_mean);
	free(conv2_batch4_std);

	// conv3_x
	for (int i = 0; i < num_filters2; i++) {
		free_tensor(&conv3_1_weights[i]);
		free_tensor(&conv3_2_weights[i]);
		free_tensor(&conv3_3_weights[i]);
		free_tensor(&conv3_4_weights[i]);
		free_tensor(&conv3_5_weights[i]);
	}

	free(conv3_batch1_beta);
	free(conv3_batch1_gamma);
	free(conv3_batch1_mean);
	free(conv3_batch1_std);

	free(conv3_batch2_beta);
	free(conv3_batch2_gamma);
	free(conv3_batch2_mean);
	free(conv3_batch2_std);

	free(conv3_batch3_beta);
	free(conv3_batch3_gamma);
	free(conv3_batch3_mean);
	free(conv3_batch3_std);

	free(conv3_batch4_beta);
	free(conv3_batch4_gamma);
	free(conv3_batch4_mean);
	free(conv3_batch4_std);

	free(conv3_batch5_beta);
	free(conv3_batch5_gamma);
	free(conv3_batch5_mean);
	free(conv3_batch5_std);

	// conv4_x
	for (int i = 0; i < num_filters2; i++) {
		free_tensor(&conv4_1_weights[i]);
		free_tensor(&conv4_2_weights[i]);
		free_tensor(&conv4_3_weights[i]);
		free_tensor(&conv4_4_weights[i]);
		free_tensor(&conv4_5_weights[i]);
	}

	free(conv4_batch1_beta);
	free(conv4_batch1_gamma);
	free(conv4_batch1_mean);
	free(conv4_batch1_std);

	free(conv4_batch2_beta);
	free(conv4_batch2_gamma);
	free(conv4_batch2_mean);
	free(conv4_batch2_std);

	free(conv4_batch3_beta);
	free(conv4_batch3_gamma);
	free(conv4_batch3_mean);
	free(conv4_batch3_std);

	free(conv4_batch4_beta);
	free(conv4_batch4_gamma);
	free(conv4_batch4_mean);
	free(conv4_batch4_std);

	free(conv4_batch5_beta);
	free(conv4_batch5_gamma);
	free(conv4_batch5_mean);
	free(conv4_batch5_std);

	// conv5_x
	for (int i = 0; i < num_filters2; i++) {
		free_tensor(&conv5_1_weights[i]);
		free_tensor(&conv5_2_weights[i]);
		free_tensor(&conv5_3_weights[i]);
		free_tensor(&conv5_4_weights[i]);
		free_tensor(&conv5_5_weights[i]);
	}

	free(conv5_batch1_beta);
	free(conv5_batch1_gamma);
	free(conv5_batch1_mean);
	free(conv5_batch1_std);

	free(conv5_batch2_beta);
	free(conv5_batch2_gamma);
	free(conv5_batch2_mean);
	free(conv5_batch2_std);

	free(conv5_batch3_beta);
	free(conv5_batch3_gamma);
	free(conv5_batch3_mean);
	free(conv5_batch3_std);

	free(conv5_batch4_beta);
	free(conv5_batch4_gamma);
	free(conv5_batch4_mean);
	free(conv5_batch4_std);

	free(conv5_batch5_beta);
	free(conv5_batch5_gamma);
	free(conv5_batch5_mean);
	free(conv5_batch5_std);

	// Free fully connected layer weights and bias
	free(fc_weights);
	free(fc_bias);
}

cudaError_t ResNetWithCuda(struct tensor* input_tensor, float* output_classes) {
	// Dimensions and sizes based on ResNet18 architecture
	int num_filters1 = 64, num_filters2 = 64, num_filters3 = 128, num_filters4 = 256, num_filters5 = 512;
	int kernel_size = 3, stride = 2, pool_size = 3, num_classes = 1000;

	// Debug
	bool debug = false;
	float* debug_data; 
	int memsize_debug;

	// Allocate memory for weights and biases
	// conv1
	struct tensor* conv1_weights;
	float* conv1_batch1_beta, * conv1_batch1_gamma, * conv1_batch1_mean, * conv1_batch1_std;

	// conv2_x
	struct tensor* conv2_1_weights, * conv2_2_weights, * conv2_3_weights, * conv2_4_weights;
	float* conv2_batch1_beta, * conv2_batch1_gamma, * conv2_batch1_mean, * conv2_batch1_std;
	float* conv2_batch2_beta, * conv2_batch2_gamma, * conv2_batch2_mean, * conv2_batch2_std;
	float* conv2_batch3_beta, * conv2_batch3_gamma, * conv2_batch3_mean, * conv2_batch3_std;
	float* conv2_batch4_beta, * conv2_batch4_gamma, * conv2_batch4_mean, * conv2_batch4_std;

	// conv3_x
	struct tensor* conv3_1_weights, * conv3_2_weights, * conv3_3_weights, * conv3_4_weights, * conv3_5_weights;
	float* conv3_batch1_beta, * conv3_batch1_gamma, * conv3_batch1_mean, * conv3_batch1_std;
	float* conv3_batch2_beta, * conv3_batch2_gamma, * conv3_batch2_mean, * conv3_batch2_std;
	float* conv3_batch3_beta, * conv3_batch3_gamma, * conv3_batch3_mean, * conv3_batch3_std;
	float* conv3_batch4_beta, * conv3_batch4_gamma, * conv3_batch4_mean, * conv3_batch4_std;
	float* conv3_batch5_beta, * conv3_batch5_gamma, * conv3_batch5_mean, * conv3_batch5_std;

	// conv4_x
	struct tensor* conv4_1_weights, * conv4_2_weights, * conv4_3_weights, * conv4_4_weights, * conv4_5_weights;
	float* conv4_batch1_beta, * conv4_batch1_gamma, * conv4_batch1_mean, * conv4_batch1_std;
	float* conv4_batch2_beta, * conv4_batch2_gamma, * conv4_batch2_mean, * conv4_batch2_std;
	float* conv4_batch3_beta, * conv4_batch3_gamma, * conv4_batch3_mean, * conv4_batch3_std;
	float* conv4_batch4_beta, * conv4_batch4_gamma, * conv4_batch4_mean, * conv4_batch4_std;
	float* conv4_batch5_beta, * conv4_batch5_gamma, * conv4_batch5_mean, * conv4_batch5_std;

	// conv5_x
	struct tensor* conv5_1_weights, * conv5_2_weights, * conv5_3_weights, * conv5_4_weights, * conv5_5_weights;
	float* conv5_batch1_beta, * conv5_batch1_gamma, * conv5_batch1_mean, * conv5_batch1_std;
	float* conv5_batch2_beta, * conv5_batch2_gamma, * conv5_batch2_mean, * conv5_batch2_std;
	float* conv5_batch3_beta, * conv5_batch3_gamma, * conv5_batch3_mean, * conv5_batch3_std;
	float* conv5_batch4_beta, * conv5_batch4_gamma, * conv5_batch4_mean, * conv5_batch4_std;
	float* conv5_batch5_beta, * conv5_batch5_gamma, * conv5_batch5_mean, * conv5_batch5_std;

	// Fully connected
	float* fc_weights, * fc_bias;

	// Load weights from binary files
	// conv1
	int memsize_conv1_weights = num_filters1 * input_tensor->depth * 7 * 7 * sizeof(float);

	conv1_weights = (struct tensor*)malloc(memsize_conv1_weights + num_filters1 * 3 * sizeof(int));
	load_conv_weights("./../../../Parameters/conv_weights_0.bin", conv1_weights, 7, input_tensor->depth, num_filters1);

	conv1_batch1_beta = (float*)malloc(num_filters1 * sizeof(float));
	load_array("./../../../Parameters/batch_beta_1.bin", conv1_batch1_beta, num_filters1);
    conv1_batch1_gamma = (float*)malloc(num_filters1 * sizeof(float));
	load_array("./../../../Parameters/batch_gamma_1.bin", conv1_batch1_gamma, num_filters1);
    conv1_batch1_mean = (float*)malloc(num_filters1 * sizeof(float));
	load_array("./../../../Parameters/batch_mean_1.bin", conv1_batch1_mean, num_filters1);
    conv1_batch1_std = (float*)malloc(num_filters1 * sizeof(float));
	load_array("./../../../Parameters/batch_std_1.bin", conv1_batch1_std, num_filters1);

	// conv2_x
	int memsize_conv2_1_weights = num_filters2 * num_filters1 * kernel_size * kernel_size * sizeof(float);
	int memsize_conv2_2_weights = num_filters2 * num_filters2 * kernel_size * kernel_size * sizeof(float);
	int memsize_conv2_3_weights = num_filters2 * num_filters2 * kernel_size * kernel_size * sizeof(float);
	int memsize_conv2_4_weights = num_filters2 * num_filters2 * kernel_size * kernel_size * sizeof(float);

	conv2_1_weights = (struct tensor*)malloc(memsize_conv2_1_weights + num_filters2 * 3 * sizeof(int));
	load_conv_weights("./../../../Parameters/conv_weights_4.bin", conv2_1_weights, kernel_size, num_filters1, num_filters2);
	conv2_2_weights = (struct tensor*)malloc(memsize_conv2_2_weights + num_filters2 * 3 * sizeof(int));
	load_conv_weights("./../../../Parameters/conv_weights_7.bin", conv2_2_weights, kernel_size, num_filters2, num_filters2);
	conv2_3_weights = (struct tensor*)malloc(memsize_conv2_3_weights + num_filters2 * 3 * sizeof(int));
	load_conv_weights("./../../../Parameters/conv_weights_9.bin", conv2_3_weights, kernel_size, num_filters2, num_filters2);
	conv2_4_weights = (struct tensor*)malloc(memsize_conv2_4_weights + num_filters2 * 3 * sizeof(int));
	load_conv_weights("./../../../Parameters/conv_weights_12.bin", conv2_4_weights, kernel_size, num_filters2, num_filters2);

	conv2_batch1_beta = (float*)malloc(num_filters2 * sizeof(float));
	load_array("./../../../Parameters/batch_beta_5.bin", conv2_batch1_beta, num_filters2);
	conv2_batch1_gamma = (float*)malloc(num_filters2 * sizeof(float));
	load_array("./../../../Parameters/batch_gamma_5.bin", conv2_batch1_gamma, num_filters2);
	conv2_batch1_mean = (float*)malloc(num_filters2 * sizeof(float));
	load_array("./../../../Parameters/batch_mean_5.bin", conv2_batch1_mean, num_filters2);
	conv2_batch1_std = (float*)malloc(num_filters2 * sizeof(float));
	load_array("./../../../Parameters/batch_std_5.bin", conv2_batch1_std, num_filters2);

	conv2_batch2_beta = (float*)malloc(num_filters2 * sizeof(float));
	load_array("./../../../Parameters/batch_beta_8.bin", conv2_batch2_beta, num_filters2);
	conv2_batch2_gamma = (float*)malloc(num_filters2 * sizeof(float));
	load_array("./../../../Parameters/batch_gamma_8.bin", conv2_batch2_gamma, num_filters2);
	conv2_batch2_mean = (float*)malloc(num_filters2 * sizeof(float));
	load_array("./../../../Parameters/batch_mean_8.bin", conv2_batch2_mean, num_filters2);
	conv2_batch2_std = (float*)malloc(num_filters2 * sizeof(float));
	load_array("./../../../Parameters/batch_std_8.bin", conv2_batch2_std, num_filters2);

	conv2_batch3_beta = (float*)malloc(num_filters2 * sizeof(float));
	load_array("./../../../Parameters/batch_beta_10.bin", conv2_batch3_beta, num_filters2);
	conv2_batch3_gamma = (float*)malloc(num_filters2 * sizeof(float));
	load_array("./../../../Parameters/batch_gamma_10.bin", conv2_batch3_gamma, num_filters2);
	conv2_batch3_mean = (float*)malloc(num_filters2 * sizeof(float));
	load_array("./../../../Parameters/batch_mean_10.bin", conv2_batch3_mean, num_filters2);
	conv2_batch3_std = (float*)malloc(num_filters2 * sizeof(float));
	load_array("./../../../Parameters/batch_std_10.bin", conv2_batch3_std, num_filters2);

	conv2_batch4_beta = (float*)malloc(num_filters2 * sizeof(float));
	load_array("./../../../Parameters/batch_beta_13.bin", conv2_batch4_beta, num_filters2);
	conv2_batch4_gamma = (float*)malloc(num_filters2 * sizeof(float));
	load_array("./../../../Parameters/batch_gamma_13.bin", conv2_batch4_gamma, num_filters2);
	conv2_batch4_mean = (float*)malloc(num_filters2 * sizeof(float));
	load_array("./../../../Parameters/batch_mean_13.bin", conv2_batch4_mean, num_filters2);
	conv2_batch4_std = (float*)malloc(num_filters2 * sizeof(float));
	load_array("./../../../Parameters/batch_std_13.bin", conv2_batch4_std, num_filters2);

	// conv3_x
	int memsize_conv3_1_weights = num_filters3 * num_filters2 * kernel_size * kernel_size * sizeof(float);
	int memsize_conv3_2_weights = num_filters3 * num_filters3 * kernel_size * kernel_size * sizeof(float);
	int memsize_conv3_3_weights = num_filters3 * num_filters2 * 1 * 1 * sizeof(float);
	int memsize_conv3_4_weights = num_filters3 * num_filters3 * kernel_size * kernel_size * sizeof(float);
	int memsize_conv3_5_weights = num_filters3 * num_filters3 * kernel_size * kernel_size * sizeof(float);

	conv3_1_weights = (struct tensor*)malloc(memsize_conv3_1_weights + num_filters3 * 3 * sizeof(int));
	load_conv_weights("./../../../Parameters/conv_weights_14.bin", conv3_1_weights, kernel_size, num_filters2, num_filters3);
	conv3_2_weights = (struct tensor*)malloc(memsize_conv3_2_weights + num_filters3 * 3 * sizeof(int));
	load_conv_weights("./../../../Parameters/conv_weights_17.bin", conv3_2_weights, kernel_size, num_filters3, num_filters3);
	conv3_3_weights = (struct tensor*)malloc(memsize_conv3_3_weights + num_filters3 * 3 * sizeof(int));
	load_conv_weights("./../../../Parameters/conv_weights_19.bin", conv3_3_weights, 1, num_filters2, num_filters3);
	conv3_4_weights = (struct tensor*)malloc(memsize_conv3_4_weights + num_filters3 * 3 * sizeof(int));
	load_conv_weights("./../../../Parameters/conv_weights_21.bin", conv3_4_weights, kernel_size, num_filters3, num_filters3);
	conv3_5_weights = (struct tensor*)malloc(memsize_conv3_5_weights + num_filters3 * 3 * sizeof(int));
	load_conv_weights("./../../../Parameters/conv_weights_24.bin", conv3_5_weights, kernel_size, num_filters3, num_filters3);

	conv3_batch1_beta = (float*)malloc(num_filters3 * sizeof(float));
	load_array("./../../../Parameters/batch_beta_15.bin", conv3_batch1_beta, num_filters3);
	conv3_batch1_gamma = (float*)malloc(num_filters3 * sizeof(float));
	load_array("./../../../Parameters/batch_gamma_15.bin", conv3_batch1_gamma, num_filters3);
	conv3_batch1_mean = (float*)malloc(num_filters3 * sizeof(float));
	load_array("./../../../Parameters/batch_mean_15.bin", conv3_batch1_mean, num_filters3);
	conv3_batch1_std = (float*)malloc(num_filters3 * sizeof(float));
	load_array("./../../../Parameters/batch_std_15.bin", conv3_batch1_std, num_filters3);

	conv3_batch2_beta = (float*)malloc(num_filters3 * sizeof(float));
	load_array("./../../../Parameters/batch_beta_18.bin", conv3_batch2_beta, num_filters3);
	conv3_batch2_gamma = (float*)malloc(num_filters3 * sizeof(float));
	load_array("./../../../Parameters/batch_gamma_18.bin", conv3_batch2_gamma, num_filters3);
	conv3_batch2_mean = (float*)malloc(num_filters3 * sizeof(float));
	load_array("./../../../Parameters/batch_mean_18.bin", conv3_batch2_mean, num_filters3);
	conv3_batch2_std = (float*)malloc(num_filters3 * sizeof(float));
	load_array("./../../../Parameters/batch_std_18.bin", conv3_batch2_std, num_filters3);

	conv3_batch3_beta = (float*)malloc(num_filters3 * sizeof(float));
	load_array("./../../../Parameters/batch_beta_20.bin", conv3_batch3_beta, num_filters3);
	conv3_batch3_gamma = (float*)malloc(num_filters3 * sizeof(float));
	load_array("./../../../Parameters/batch_gamma_20.bin", conv3_batch3_gamma, num_filters3);
	conv3_batch3_mean = (float*)malloc(num_filters3 * sizeof(float));
	load_array("./../../../Parameters/batch_mean_20.bin", conv3_batch3_mean, num_filters3);
	conv3_batch3_std = (float*)malloc(num_filters3 * sizeof(float));
	load_array("./../../../Parameters/batch_std_20.bin", conv3_batch3_std, num_filters3);

	conv3_batch4_beta = (float*)malloc(num_filters3 * sizeof(float));
	load_array("./../../../Parameters/batch_beta_22.bin", conv3_batch4_beta, num_filters3);
	conv3_batch4_gamma = (float*)malloc(num_filters3 * sizeof(float));
	load_array("./../../../Parameters/batch_gamma_22.bin", conv3_batch4_gamma, num_filters3);
	conv3_batch4_mean = (float*)malloc(num_filters3 * sizeof(float));
	load_array("./../../../Parameters/batch_mean_22.bin", conv3_batch4_mean, num_filters3);
	conv3_batch4_std = (float*)malloc(num_filters3 * sizeof(float));
	load_array("./../../../Parameters/batch_std_22.bin", conv3_batch4_std, num_filters3);

	conv3_batch5_beta = (float*)malloc(num_filters3 * sizeof(float));
	load_array("./../../../Parameters/batch_beta_25.bin", conv3_batch5_beta, num_filters3);
	conv3_batch5_gamma = (float*)malloc(num_filters3 * sizeof(float));
	load_array("./../../../Parameters/batch_gamma_25.bin", conv3_batch5_gamma, num_filters3);
	conv3_batch5_mean = (float*)malloc(num_filters3 * sizeof(float));
	load_array("./../../../Parameters/batch_mean_25.bin", conv3_batch5_mean, num_filters3);
	conv3_batch5_std = (float*)malloc(num_filters3 * sizeof(float));
	load_array("./../../../Parameters/batch_std_25.bin", conv3_batch5_std, num_filters3);

	// conv4_x
	int memsize_conv4_1_weights = num_filters4 * num_filters3 * kernel_size * kernel_size * sizeof(float);
	int memsize_conv4_2_weights = num_filters4 * num_filters4 * kernel_size * kernel_size * sizeof(float);
	int memsize_conv4_3_weights = num_filters4 * num_filters3 * 1 * 1 * sizeof(float);
	int memsize_conv4_4_weights = num_filters4 * num_filters4 * kernel_size * kernel_size * sizeof(float);
	int memsize_conv4_5_weights = num_filters4 * num_filters4 * kernel_size * kernel_size * sizeof(float);

	conv4_1_weights = (struct tensor*)malloc(memsize_conv4_1_weights + num_filters4 * 3 * sizeof(int));
	load_conv_weights("./../../../Parameters/conv_weights_26.bin", conv4_1_weights, kernel_size, num_filters3, num_filters4);
	conv4_2_weights = (struct tensor*)malloc(memsize_conv4_2_weights + num_filters4 * 3 * sizeof(int));
	load_conv_weights("./../../../Parameters/conv_weights_29.bin", conv4_2_weights, kernel_size, num_filters4, num_filters4);
	conv4_3_weights = (struct tensor*)malloc(memsize_conv4_3_weights + num_filters4 * 3 * sizeof(int));
	load_conv_weights("./../../../Parameters/conv_weights_31.bin", conv4_3_weights, 1, num_filters3, num_filters4);
	conv4_4_weights = (struct tensor*)malloc(memsize_conv4_4_weights + num_filters4 * 3 * sizeof(int));
	load_conv_weights("./../../../Parameters/conv_weights_33.bin", conv4_4_weights, kernel_size, num_filters4, num_filters4);
	conv4_5_weights = (struct tensor*)malloc(memsize_conv4_5_weights + num_filters4 * 3 * sizeof(int));
	load_conv_weights("./../../../Parameters/conv_weights_36.bin", conv4_5_weights, kernel_size, num_filters4, num_filters4);

	conv4_batch1_beta = (float*)malloc(num_filters4 * sizeof(float));
	load_array("./../../../Parameters/batch_beta_27.bin", conv4_batch1_beta, num_filters4);
	conv4_batch1_gamma = (float*)malloc(num_filters4 * sizeof(float));
	load_array("./../../../Parameters/batch_gamma_27.bin", conv4_batch1_gamma, num_filters4);
	conv4_batch1_mean = (float*)malloc(num_filters4 * sizeof(float));
	load_array("./../../../Parameters/batch_mean_27.bin", conv4_batch1_mean, num_filters4);
	conv4_batch1_std = (float*)malloc(num_filters4 * sizeof(float));
	load_array("./../../../Parameters/batch_std_27.bin", conv4_batch1_std, num_filters4);

	conv4_batch2_beta = (float*)malloc(num_filters4 * sizeof(float));
	load_array("./../../../Parameters/batch_beta_30.bin", conv4_batch2_beta, num_filters4);
	conv4_batch2_gamma = (float*)malloc(num_filters4 * sizeof(float));
	load_array("./../../../Parameters/batch_gamma_30.bin", conv4_batch2_gamma, num_filters4);
	conv4_batch2_mean = (float*)malloc(num_filters4 * sizeof(float));
	load_array("./../../../Parameters/batch_mean_30.bin", conv4_batch2_mean, num_filters4);
	conv4_batch2_std = (float*)malloc(num_filters4 * sizeof(float));
	load_array("./../../../Parameters/batch_std_30.bin", conv4_batch2_std, num_filters4);

	conv4_batch3_beta = (float*)malloc(num_filters4 * sizeof(float));
	load_array("./../../../Parameters/batch_beta_32.bin", conv4_batch3_beta, num_filters4);
	conv4_batch3_gamma = (float*)malloc(num_filters4 * sizeof(float));
	load_array("./../../../Parameters/batch_gamma_32.bin", conv4_batch3_gamma, num_filters4);
	conv4_batch3_mean = (float*)malloc(num_filters4 * sizeof(float));
	load_array("./../../../Parameters/batch_mean_32.bin", conv4_batch3_mean, num_filters4);
	conv4_batch3_std = (float*)malloc(num_filters4 * sizeof(float));
	load_array("./../../../Parameters/batch_std_32.bin", conv4_batch3_std, num_filters4);

	conv4_batch4_beta = (float*)malloc(num_filters4 * sizeof(float));
	load_array("./../../../Parameters/batch_beta_34.bin", conv4_batch4_beta, num_filters4);
	conv4_batch4_gamma = (float*)malloc(num_filters4 * sizeof(float));
	load_array("./../../../Parameters/batch_gamma_34.bin", conv4_batch4_gamma, num_filters4);
	conv4_batch4_mean = (float*)malloc(num_filters4 * sizeof(float));
	load_array("./../../../Parameters/batch_mean_34.bin", conv4_batch4_mean, num_filters4);
	conv4_batch4_std = (float*)malloc(num_filters4 * sizeof(float));
	load_array("./../../../Parameters/batch_std_34.bin", conv4_batch4_std, num_filters4);

	conv4_batch5_beta = (float*)malloc(num_filters4 * sizeof(float));
	load_array("./../../../Parameters/batch_beta_37.bin", conv4_batch5_beta, num_filters4);
	conv4_batch5_gamma = (float*)malloc(num_filters4 * sizeof(float));
	load_array("./../../../Parameters/batch_gamma_37.bin", conv4_batch5_gamma, num_filters4);
	conv4_batch5_mean = (float*)malloc(num_filters4 * sizeof(float));
	load_array("./../../../Parameters/batch_mean_37.bin", conv4_batch5_mean, num_filters4);
	conv4_batch5_std = (float*)malloc(num_filters4 * sizeof(float));
	load_array("./../../../Parameters/batch_std_37.bin", conv4_batch5_std, num_filters4);

	// conv5_x
	int memsize_conv5_1_weights = num_filters5 * num_filters4 * kernel_size * kernel_size * sizeof(float);
	int memsize_conv5_2_weights = num_filters5 * num_filters5 * kernel_size * kernel_size * sizeof(float);
	int memsize_conv5_3_weights = num_filters5 * num_filters4 * 1 * 1 * sizeof(float);
	int memsize_conv5_4_weights = num_filters5 * num_filters5 * kernel_size * kernel_size * sizeof(float);
	int memsize_conv5_5_weights = num_filters5 * num_filters5 * kernel_size * kernel_size * sizeof(float);

	conv5_1_weights = (struct tensor*)malloc(memsize_conv5_1_weights + num_filters5 * 3 * sizeof(int));
	load_conv_weights("./../../../Parameters/conv_weights_38.bin", conv5_1_weights, kernel_size, num_filters4, num_filters5);
	conv5_2_weights = (struct tensor*)malloc(memsize_conv5_2_weights + num_filters5 * 3 * sizeof(int));
	load_conv_weights("./../../../Parameters/conv_weights_41.bin", conv5_2_weights, kernel_size, num_filters5, num_filters5);
	conv5_3_weights = (struct tensor*)malloc(memsize_conv5_3_weights + num_filters5 * 3 * sizeof(int));
	load_conv_weights("./../../../Parameters/conv_weights_43.bin", conv5_3_weights, 1, num_filters4, num_filters5);
	conv5_4_weights = (struct tensor*)malloc(memsize_conv5_4_weights + num_filters5 * 3 * sizeof(int));
	load_conv_weights("./../../../Parameters/conv_weights_45.bin", conv5_4_weights, kernel_size, num_filters5, num_filters5);
	conv5_5_weights = (struct tensor*)malloc(memsize_conv5_5_weights + num_filters5 * 3 * sizeof(int));
	load_conv_weights("./../../../Parameters/conv_weights_48.bin", conv5_5_weights, kernel_size, num_filters5, num_filters5);
	
	conv5_batch1_beta = (float*)malloc(num_filters5 * sizeof(float));
	load_array("./../../../Parameters/batch_beta_39.bin", conv5_batch1_beta, num_filters5);
	conv5_batch1_gamma = (float*)malloc(num_filters5 * sizeof(float));
	load_array("./../../../Parameters/batch_gamma_39.bin", conv5_batch1_gamma, num_filters5);
	conv5_batch1_mean = (float*)malloc(num_filters5 * sizeof(float));
	load_array("./../../../Parameters/batch_mean_39.bin", conv5_batch1_mean, num_filters5);
	conv5_batch1_std = (float*)malloc(num_filters5 * sizeof(float));
	load_array("./../../../Parameters/batch_std_39.bin", conv5_batch1_std, num_filters5);

	conv5_batch2_beta = (float*)malloc(num_filters5 * sizeof(float));
	load_array("./../../../Parameters/batch_beta_42.bin", conv5_batch2_beta, num_filters5);
	conv5_batch2_gamma = (float*)malloc(num_filters5 * sizeof(float));
	load_array("./../../../Parameters/batch_gamma_42.bin", conv5_batch2_gamma, num_filters5);
	conv5_batch2_mean = (float*)malloc(num_filters5 * sizeof(float));
	load_array("./../../../Parameters/batch_mean_42.bin", conv5_batch2_mean, num_filters5);
	conv5_batch2_std = (float*)malloc(num_filters5 * sizeof(float));
	load_array("./../../../Parameters/batch_std_42.bin", conv5_batch2_std, num_filters5);

	conv5_batch3_beta = (float*)malloc(num_filters5 * sizeof(float));
	load_array("./../../../Parameters/batch_beta_44.bin", conv5_batch3_beta, num_filters5);
	conv5_batch3_gamma = (float*)malloc(num_filters5 * sizeof(float));
	load_array("./../../../Parameters/batch_gamma_44.bin", conv5_batch3_gamma, num_filters5);
	conv5_batch3_mean = (float*)malloc(num_filters5 * sizeof(float));
	load_array("./../../../Parameters/batch_mean_44.bin", conv5_batch3_mean, num_filters5);
	conv5_batch3_std = (float*)malloc(num_filters5 * sizeof(float));
	load_array("./../../../Parameters/batch_std_44.bin", conv5_batch3_std, num_filters5);

	conv5_batch4_beta = (float*)malloc(num_filters5 * sizeof(float));
	load_array("./../../../Parameters/batch_beta_46.bin", conv5_batch4_beta, num_filters5);
	conv5_batch4_gamma = (float*)malloc(num_filters5 * sizeof(float));
	load_array("./../../../Parameters/batch_gamma_46.bin", conv5_batch4_gamma, num_filters5);
	conv5_batch4_mean = (float*)malloc(num_filters5 * sizeof(float));
	load_array("./../../../Parameters/batch_mean_46.bin", conv5_batch4_mean, num_filters5);
	conv5_batch4_std = (float*)malloc(num_filters5 * sizeof(float));
	load_array("./../../../Parameters/batch_std_46.bin", conv5_batch4_std, num_filters5);

	conv5_batch5_beta = (float*)malloc(num_filters5 * sizeof(float));
	load_array("./../../../Parameters/batch_beta_49.bin", conv5_batch5_beta, num_filters5);
	conv5_batch5_gamma = (float*)malloc(num_filters5 * sizeof(float));
	load_array("./../../../Parameters/batch_gamma_49.bin", conv5_batch5_gamma, num_filters5);
	conv5_batch5_mean = (float*)malloc(num_filters5 * sizeof(float));
	load_array("./../../../Parameters/batch_mean_49.bin", conv5_batch5_mean, num_filters5);
	conv5_batch5_std = (float*)malloc(num_filters5 * sizeof(float));
	load_array("./../../../Parameters/batch_std_49.bin", conv5_batch5_std, num_filters5);

	// Fully connected
	fc_weights = (float*)malloc(num_filters5 * num_classes * sizeof(float)); 
	load_matrix("./../../../Parameters/linear_weights_51.bin", fc_weights, num_filters5, num_classes);
	fc_bias = (float*)malloc(num_classes * sizeof(float));
	load_array("./../../../Parameters/linear_bias_51.bin", fc_bias, num_classes);

	// Allocate device (GPU) memory for input, output, and intermediate tensors
	// Device memory for input data
	float* dev_input_data;
	int memsize_input_data = input_tensor->depth * input_tensor->row * input_tensor->col * sizeof(float);
	cudaMalloc((void**)&dev_input_data, memsize_input_data);
	cudaMemcpy(dev_input_data, input_tensor->data, memsize_input_data, cudaMemcpyHostToDevice);

	// Device memory for output data
	float* dev_output_classes;
	cudaMalloc((void**)&dev_output_classes, num_classes * sizeof(float));

	// Network architecture
	//=================================
	//							 Layer 1 
	//=================================
	// Define the hyperparameters
	int image_size = input_tensor->col;
	int num_channel = input_tensor->depth;

	// Allocate device (GPU) memory for kernels and intermediate tensors
	// Device memory for conv1_weights
	float* dev_conv1_weights;
	cudaMalloc((void**)&dev_conv1_weights, memsize_conv1_weights);
	for (int i = 0; i < num_filters1; i++) {
		cudaMemcpy(dev_conv1_weights + i * 7 * 7 * num_channel, conv1_weights[i].data, 7 * 7 * num_channel * sizeof(float), cudaMemcpyHostToDevice);
	}

	// Device memory for BatchNorm
	int memsize_conv1_batch1 = num_filters1 * sizeof(float);
	float* dev_conv1_batch1_beta;
	cudaMalloc((void**)&dev_conv1_batch1_beta, memsize_conv1_batch1);
	cudaMemcpy(dev_conv1_batch1_beta, conv1_batch1_beta, memsize_conv1_batch1, cudaMemcpyHostToDevice);

	float* dev_conv1_batch1_gamma;
	cudaMalloc((void**)&dev_conv1_batch1_gamma, memsize_conv1_batch1);
	cudaMemcpy(dev_conv1_batch1_gamma, conv1_batch1_gamma, memsize_conv1_batch1, cudaMemcpyHostToDevice);

	float* dev_conv1_batch1_mean;
	cudaMalloc((void**)&dev_conv1_batch1_mean, memsize_conv1_batch1);
	cudaMemcpy(dev_conv1_batch1_mean, conv1_batch1_mean, memsize_conv1_batch1, cudaMemcpyHostToDevice);

	float* dev_conv1_batch1_std;
	cudaMalloc((void**)&dev_conv1_batch1_std, memsize_conv1_batch1);
	cudaMemcpy(dev_conv1_batch1_std, conv1_batch1_std, memsize_conv1_batch1, cudaMemcpyHostToDevice);

	// Device memory for intermediate tensors
	int memsize_output_data = num_filters1 * (image_size / 2) * (image_size / 2) * sizeof(float);
	float* dev_layer1_output_data1;
	cudaMalloc((void**)&dev_layer1_output_data1, memsize_output_data);

	float* dev_layer1_output_data2;
	cudaMalloc((void**)&dev_layer1_output_data2, memsize_output_data);

	// Intermediate tensors for residual connections
	float* dev_layer1_output_data3;
	cudaMalloc((void**)&dev_layer1_output_data3, memsize_output_data);
	
	// Define CudaKernel settings
	dim3 threadInBlock(8, 8, 16); // Adjust to suitable block size
	dim3 numBlocks;
	numBlocks.x = (image_size + threadInBlock.x - 1) / threadInBlock.x;
	numBlocks.y = (image_size + threadInBlock.y - 1) / threadInBlock.y;
	numBlocks.z = num_filters1;
	int memsize_shared_memory = threadInBlock.x * threadInBlock.y * threadInBlock.z * sizeof(float);
	
	// conv1
	convolution_parallel << <numBlocks, threadInBlock, memsize_shared_memory >> > (dev_input_data, image_size, image_size, num_channel, dev_conv1_weights, 7, stride, dev_layer1_output_data1);

	// Define CudaKernel settings
	image_size = image_size/2;
	threadInBlock.x = 16; 
	threadInBlock.y = 16;
	threadInBlock.z = 1;
	numBlocks.x = (image_size + threadInBlock.x - 1) / threadInBlock.x;
	numBlocks.y = (image_size + threadInBlock.y - 1) / threadInBlock.y;
	numBlocks.z = num_filters1;

	// BatchNorm
	batch_normalization_parallel << <numBlocks, threadInBlock >> > (dev_layer1_output_data1, image_size, image_size, dev_conv1_batch1_beta, dev_conv1_batch1_gamma, dev_conv1_batch1_mean, dev_conv1_batch1_std, dev_layer1_output_data2);

	// ReLU
	relu_parallel << <numBlocks, threadInBlock >> > (dev_layer1_output_data2, image_size, image_size, dev_layer1_output_data1);

	// Debug
	if (debug) {
		memsize_debug = image_size * image_size * num_filters1 * sizeof(float);
		debug_data = (float*)malloc(memsize_debug);
		cudaMemcpy(debug_data, dev_layer1_output_data1, memsize_debug, cudaMemcpyDeviceToHost);
		float max = FLT_MIN;
		float min = FLT_MAX;
		for (int i = 0; i < image_size * image_size * num_filters1; i++) {
			if (debug_data[i] > max) {
				max = debug_data[i];
			}
			if (debug_data[i] < min) {
				min = debug_data[i];
			}
		}
		printf("Output layer 1:\n");
		printf("Max value: %f\n", max);
		printf("Min value: %f\n", min);
	}

	//=================================
	//							 Layer 2 
	//=================================
	// Allocate device (GPU) memory for kernels and intermediate tensors
	// Device memory for conv2_x_weights
	float* dev_conv2_1_weights;
	cudaMalloc((void**)&dev_conv2_1_weights, memsize_conv2_1_weights);

	float* dev_conv2_2_weights;
	cudaMalloc((void**)&dev_conv2_2_weights, memsize_conv2_2_weights);

	float* dev_conv2_3_weights;
	cudaMalloc((void**)&dev_conv2_3_weights, memsize_conv2_3_weights);

	float* dev_conv2_4_weights;
	cudaMalloc((void**)&dev_conv2_4_weights, memsize_conv2_4_weights);

	for (int i = 0; i < num_filters2; i++) {
		cudaMemcpy(dev_conv2_1_weights + i * kernel_size * kernel_size * num_filters1, conv2_1_weights[i].data, kernel_size * kernel_size * num_filters1 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_conv2_2_weights + i * kernel_size * kernel_size * num_filters2, conv2_2_weights[i].data, kernel_size * kernel_size * num_filters2 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_conv2_3_weights + i * kernel_size * kernel_size * num_filters2, conv2_3_weights[i].data, kernel_size * kernel_size * num_filters2 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_conv2_4_weights + i * kernel_size * kernel_size * num_filters2, conv2_4_weights[i].data, kernel_size * kernel_size * num_filters2 * sizeof(float), cudaMemcpyHostToDevice);
	}

	// Device memory for BatchNorm
	int memsize_conv2_batch = num_filters2 * sizeof(float);
	// conv2_batch1
	float* dev_conv2_batch1_beta;
	cudaMalloc((void**)&dev_conv2_batch1_beta, memsize_conv2_batch);
	cudaMemcpy(dev_conv2_batch1_beta, conv2_batch1_beta, memsize_conv2_batch, cudaMemcpyHostToDevice);

	float* dev_conv2_batch1_gamma;
	cudaMalloc((void**)&dev_conv2_batch1_gamma, memsize_conv2_batch);
	cudaMemcpy(dev_conv2_batch1_gamma, conv2_batch1_gamma, memsize_conv2_batch, cudaMemcpyHostToDevice);

	float* dev_conv2_batch1_mean;
	cudaMalloc((void**)&dev_conv2_batch1_mean, memsize_conv2_batch);
	cudaMemcpy(dev_conv2_batch1_mean, conv2_batch1_mean, memsize_conv2_batch, cudaMemcpyHostToDevice);

	float* dev_conv2_batch1_std;
	cudaMalloc((void**)&dev_conv2_batch1_std, memsize_conv2_batch);
	cudaMemcpy(dev_conv2_batch1_std, conv2_batch1_std, memsize_conv2_batch, cudaMemcpyHostToDevice);

	// conv2_batch2
	float* dev_conv2_batch2_beta;
	cudaMalloc((void**)&dev_conv2_batch2_beta, memsize_conv2_batch);
	cudaMemcpy(dev_conv2_batch2_beta, conv2_batch2_beta, memsize_conv2_batch, cudaMemcpyHostToDevice);

	float* dev_conv2_batch2_gamma;
	cudaMalloc((void**)&dev_conv2_batch2_gamma, memsize_conv2_batch);
	cudaMemcpy(dev_conv2_batch2_gamma, conv2_batch2_gamma, memsize_conv2_batch, cudaMemcpyHostToDevice);

	float* dev_conv2_batch2_mean;
	cudaMalloc((void**)&dev_conv2_batch2_mean, memsize_conv2_batch);
	cudaMemcpy(dev_conv2_batch2_mean, conv2_batch2_mean, memsize_conv2_batch, cudaMemcpyHostToDevice);

	float* dev_conv2_batch2_std;
	cudaMalloc((void**)&dev_conv2_batch2_std, memsize_conv2_batch);
	cudaMemcpy(dev_conv2_batch2_std, conv2_batch2_std, memsize_conv2_batch, cudaMemcpyHostToDevice);

	// conv2_batch3
	float* dev_conv2_batch3_beta;
	cudaMalloc((void**)&dev_conv2_batch3_beta, memsize_conv2_batch);
	cudaMemcpy(dev_conv2_batch3_beta, conv2_batch3_beta, memsize_conv2_batch, cudaMemcpyHostToDevice);

	float* dev_conv2_batch3_gamma;
	cudaMalloc((void**)&dev_conv2_batch3_gamma, memsize_conv2_batch);
	cudaMemcpy(dev_conv2_batch3_gamma, conv2_batch3_gamma, memsize_conv2_batch, cudaMemcpyHostToDevice);

	float* dev_conv2_batch3_mean;
	cudaMalloc((void**)&dev_conv2_batch3_mean, memsize_conv2_batch);
	cudaMemcpy(dev_conv2_batch3_mean, conv2_batch3_mean, memsize_conv2_batch, cudaMemcpyHostToDevice);

	float* dev_conv2_batch3_std;
	cudaMalloc((void**)&dev_conv2_batch3_std, memsize_conv2_batch);
	cudaMemcpy(dev_conv2_batch3_std, conv2_batch3_std, memsize_conv2_batch, cudaMemcpyHostToDevice);

	// conv2_batch4
	float* dev_conv2_batch4_beta;
	cudaMalloc((void**)&dev_conv2_batch4_beta, memsize_conv2_batch);
	cudaMemcpy(dev_conv2_batch4_beta, conv2_batch4_beta, memsize_conv2_batch, cudaMemcpyHostToDevice);

	float* dev_conv2_batch4_gamma;
	cudaMalloc((void**)&dev_conv2_batch4_gamma, memsize_conv2_batch);
	cudaMemcpy(dev_conv2_batch4_gamma, conv2_batch4_gamma, memsize_conv2_batch, cudaMemcpyHostToDevice);

	float* dev_conv2_batch4_mean;
	cudaMalloc((void**)&dev_conv2_batch4_mean, memsize_conv2_batch);
	cudaMemcpy(dev_conv2_batch4_mean, conv2_batch4_mean, memsize_conv2_batch, cudaMemcpyHostToDevice);

	float* dev_conv2_batch4_std;
	cudaMalloc((void**)&dev_conv2_batch4_std, memsize_conv2_batch);
	cudaMemcpy(dev_conv2_batch4_std, conv2_batch4_std, memsize_conv2_batch, cudaMemcpyHostToDevice);

	// Device memory for intermediate tensors
	memsize_output_data = num_filters2 * (image_size/2) * (image_size/2) * sizeof(float);
	float* dev_layer2_output_data1;
	cudaMalloc((void**)&dev_layer2_output_data1, memsize_output_data);

	float* dev_layer2_output_data2;
	cudaMalloc((void**)&dev_layer2_output_data2, memsize_output_data);

	// Intermediate tensors for residual connections
	float* dev_layer2_output_data3;
	cudaMalloc((void**)&dev_layer2_output_data3, memsize_output_data);

	// MaxPool
	max_pooling_parallel << <numBlocks, threadInBlock >> > (dev_layer1_output_data1, image_size, image_size, num_filters1, pool_size, stride, dev_layer2_output_data1);

	// Identity block 1
	// Define CudaKernel settings
	image_size = image_size / 2;
	threadInBlock.x = 16;
	threadInBlock.y = 16;
	threadInBlock.z = 1;
	numBlocks.x = (image_size + threadInBlock.x - 1) / threadInBlock.x;
	numBlocks.y = (image_size + threadInBlock.y - 1) / threadInBlock.y;
	numBlocks.z = num_filters2;

	// ReLU (residual connection)
	relu_parallel << <numBlocks, threadInBlock >> > (dev_layer2_output_data1, image_size, image_size, dev_layer2_output_data3);

	// Define CudaKernel settings
	threadInBlock.x = 8;
	threadInBlock.y = 8;
	threadInBlock.z = 16;
	numBlocks.x = (image_size + threadInBlock.x - 1) / threadInBlock.x;
	numBlocks.y = (image_size + threadInBlock.y - 1) / threadInBlock.y;
	numBlocks.z = num_filters2;
	memsize_shared_memory = threadInBlock.x * threadInBlock.y * threadInBlock.z * sizeof(float);

	// conv2_1
	convolution_parallel << <numBlocks, threadInBlock, memsize_shared_memory >> > (dev_layer2_output_data1, image_size, image_size, num_filters1, dev_conv2_1_weights, kernel_size, 1, dev_layer2_output_data2);

	// Define CudaKernel settings
	threadInBlock.x = 16;
	threadInBlock.y = 16;
	threadInBlock.z = 1;
	numBlocks.x = (image_size + threadInBlock.x - 1) / threadInBlock.x;
	numBlocks.y = (image_size + threadInBlock.y - 1) / threadInBlock.y;
	numBlocks.z = num_filters2;

	// BatchNorm
	batch_normalization_parallel << <numBlocks, threadInBlock >> > (dev_layer2_output_data2, image_size, image_size, dev_conv2_batch1_beta, dev_conv2_batch1_gamma, dev_conv2_batch1_mean, dev_conv2_batch1_std, dev_layer2_output_data1);

	// ReLU
	relu_parallel << <numBlocks, threadInBlock >> > (dev_layer2_output_data1, image_size, image_size, dev_layer2_output_data2);

	// Define CudaKernel settings
	threadInBlock.x = 8;
	threadInBlock.y = 8;
	threadInBlock.z = 16;
	numBlocks.x = (image_size + threadInBlock.x - 1) / threadInBlock.x;
	numBlocks.y = (image_size + threadInBlock.y - 1) / threadInBlock.y;
	numBlocks.z = num_filters2;
	memsize_shared_memory = threadInBlock.x * threadInBlock.y * threadInBlock.z * sizeof(float);

	// conv2_2
	convolution_parallel << <numBlocks, threadInBlock, memsize_shared_memory >> > (dev_layer2_output_data2, image_size, image_size, num_filters2, dev_conv2_2_weights, kernel_size, 1, dev_layer2_output_data1);

	// Define CudaKernel settings
	threadInBlock.x = 16;
	threadInBlock.y = 16;
	threadInBlock.z = 1;
	numBlocks.x = (image_size + threadInBlock.x - 1) / threadInBlock.x;
	numBlocks.y = (image_size + threadInBlock.y - 1) / threadInBlock.y;
	numBlocks.z = num_filters2;

	// BatchNorm
	batch_normalization_parallel << <numBlocks, threadInBlock >> > (dev_layer2_output_data1, image_size, image_size, dev_conv2_batch2_beta, dev_conv2_batch2_gamma, dev_conv2_batch2_mean, dev_conv2_batch2_std, dev_layer2_output_data2);

	// ResidualConnection
	add_tensors_parallel << <numBlocks, threadInBlock >> > (dev_layer2_output_data2, dev_layer2_output_data3, image_size, image_size, dev_layer2_output_data1);

	// ReLU
	relu_parallel << <numBlocks, threadInBlock >> > (dev_layer2_output_data1, image_size, image_size, dev_layer2_output_data2);

	// Debug
	if (debug) {
		memsize_debug = image_size * image_size * num_filters2 * sizeof(float);
		debug_data = (float*)malloc(memsize_debug);
		cudaMemcpy(debug_data, dev_layer2_output_data2, memsize_debug, cudaMemcpyDeviceToHost);
		float max = FLT_MIN;
		float min = FLT_MAX;
		for (int i = 0; i < image_size * image_size * num_filters2; i++) {
			if (debug_data[i] > max) {
				max = debug_data[i];
			}
			if (debug_data[i] < min) {
				min = debug_data[i];
			}
		}
		printf("Output layer 2 (Identity block 1):\n");
		printf("Max value: %f\n", max);
		printf("Min value: %f\n", min);
	}

	// Identity block 2
	// ReLU (residual connection)
	relu_parallel << <numBlocks, threadInBlock >> > (dev_layer2_output_data2, image_size, image_size, dev_layer2_output_data3);

	// Define CudaKernel settings
	threadInBlock.x = 8;
	threadInBlock.y = 8;
	threadInBlock.z = 16;
	numBlocks.x = (image_size + threadInBlock.x - 1) / threadInBlock.x;
	numBlocks.y = (image_size + threadInBlock.y - 1) / threadInBlock.y;
	numBlocks.z = num_filters2;
	memsize_shared_memory = threadInBlock.x * threadInBlock.y * threadInBlock.z * sizeof(float);

	// conv2_3
	convolution_parallel << <numBlocks, threadInBlock, memsize_shared_memory >> > (dev_layer2_output_data2, image_size, image_size, num_filters2, dev_conv2_3_weights, kernel_size, 1, dev_layer2_output_data1);

	// Define CudaKernel settings
	threadInBlock.x = 16;
	threadInBlock.y = 16;
	threadInBlock.z = 1;
	numBlocks.x = (image_size + threadInBlock.x - 1) / threadInBlock.x;
	numBlocks.y = (image_size + threadInBlock.y - 1) / threadInBlock.y;
	numBlocks.z = num_filters2;

	// BatchNorm
	batch_normalization_parallel << <numBlocks, threadInBlock >> > (dev_layer2_output_data1, image_size, image_size, dev_conv2_batch3_beta, dev_conv2_batch3_gamma, dev_conv2_batch3_mean, dev_conv2_batch3_std, dev_layer2_output_data2);

	// ReLU
	relu_parallel << <numBlocks, threadInBlock >> > (dev_layer2_output_data2, image_size, image_size, dev_layer2_output_data1);

	threadInBlock.x = 8;
	threadInBlock.y = 8;
	threadInBlock.z = 16;
	numBlocks.x = (image_size + threadInBlock.x - 1) / threadInBlock.x;
	numBlocks.y = (image_size + threadInBlock.y - 1) / threadInBlock.y;
	numBlocks.z = num_filters2;
	memsize_shared_memory = threadInBlock.x * threadInBlock.y * threadInBlock.z * sizeof(float);

	// conv2_4
	convolution_parallel << <numBlocks, threadInBlock, memsize_shared_memory >> > (dev_layer2_output_data1, image_size, image_size, num_filters2, dev_conv2_4_weights, kernel_size, 1, dev_layer2_output_data2);

	// Define CudaKernel settings
	threadInBlock.x = 16;
	threadInBlock.y = 16;
	threadInBlock.z = 1;
	numBlocks.x = (image_size + threadInBlock.x - 1) / threadInBlock.x;
	numBlocks.y = (image_size + threadInBlock.y - 1) / threadInBlock.y;
	numBlocks.z = num_filters2;

	// BatchNorm
	batch_normalization_parallel << <numBlocks, threadInBlock >> > (dev_layer2_output_data2, image_size, image_size, dev_conv2_batch4_beta, dev_conv2_batch4_gamma, dev_conv2_batch4_mean, dev_conv2_batch4_std, dev_layer2_output_data1);

	// ResidualConnection
	add_tensors_parallel << <numBlocks, threadInBlock >> > (dev_layer2_output_data1, dev_layer2_output_data3, image_size, image_size, dev_layer2_output_data2);

	// ReLU
	relu_parallel << <numBlocks, threadInBlock >> > (dev_layer2_output_data2, image_size, image_size, dev_layer2_output_data1);

	// Debug
	if (debug) {
		memsize_debug = image_size * image_size * num_filters2 * sizeof(float);
		debug_data = (float*)malloc(memsize_debug);
		cudaMemcpy(debug_data, dev_layer2_output_data1, memsize_debug, cudaMemcpyDeviceToHost);
		float max = FLT_MIN;
		float min = FLT_MAX;
		for (int i = 0; i < image_size * image_size * num_filters2; i++) {
			if (debug_data[i] > max) {
				max = debug_data[i];
			}
			if (debug_data[i] < min) {
				min = debug_data[i];
			}
		}
		printf("Output layer 2 (Identity block 2):\n");
		printf("Max value: %f\n", max);
		printf("Min value: %f\n", min);
	}

	//=================================
	//							 Layer 3 
	//=================================
	// Allocate device (GPU) memory for kernels and intermediate tensors
	// Device memory for conv3_x_weights
	float* dev_conv3_1_weights;
	cudaMalloc((void**)&dev_conv3_1_weights, memsize_conv3_1_weights);

	float* dev_conv3_2_weights;
	cudaMalloc((void**)&dev_conv3_2_weights, memsize_conv3_2_weights);

	float* dev_conv3_3_weights;
	cudaMalloc((void**)&dev_conv3_3_weights, memsize_conv3_3_weights);

	float* dev_conv3_4_weights;
	cudaMalloc((void**)&dev_conv3_4_weights, memsize_conv3_4_weights);

	float* dev_conv3_5_weights;
	cudaMalloc((void**)&dev_conv3_5_weights, memsize_conv3_5_weights);

	for (int i = 0; i < num_filters3; i++) {
		cudaMemcpy(dev_conv3_1_weights + i * kernel_size * kernel_size * num_filters2, conv3_1_weights[i].data, kernel_size * kernel_size * num_filters2 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_conv3_2_weights + i * kernel_size * kernel_size * num_filters3, conv3_2_weights[i].data, kernel_size * kernel_size * num_filters3 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_conv3_3_weights + i * 1 * 1 * num_filters2, conv3_3_weights[i].data, 1 * 1 * num_filters2 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_conv3_4_weights + i * kernel_size * kernel_size * num_filters3, conv3_4_weights[i].data, kernel_size * kernel_size * num_filters3 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_conv3_5_weights + i * kernel_size * kernel_size * num_filters3, conv3_5_weights[i].data, kernel_size * kernel_size * num_filters3 * sizeof(float), cudaMemcpyHostToDevice);
	}

	// Device memory for BatchNorm
	int memsize_conv3_batch = num_filters3 * sizeof(float);
	// conv3_batch1
	float* dev_conv3_batch1_beta;
	cudaMalloc((void**)&dev_conv3_batch1_beta, memsize_conv3_batch);
	cudaMemcpy(dev_conv3_batch1_beta, conv3_batch1_beta, memsize_conv3_batch, cudaMemcpyHostToDevice);

	float* dev_conv3_batch1_gamma;
	cudaMalloc((void**)&dev_conv3_batch1_gamma, memsize_conv3_batch);
	cudaMemcpy(dev_conv3_batch1_gamma, conv3_batch1_gamma, memsize_conv3_batch, cudaMemcpyHostToDevice);

	float* dev_conv3_batch1_mean;
	cudaMalloc((void**)&dev_conv3_batch1_mean, memsize_conv3_batch);
	cudaMemcpy(dev_conv3_batch1_mean, conv3_batch1_mean, memsize_conv3_batch, cudaMemcpyHostToDevice);

	float* dev_conv3_batch1_std;
	cudaMalloc((void**)&dev_conv3_batch1_std, memsize_conv3_batch);
	cudaMemcpy(dev_conv3_batch1_std, conv3_batch1_std, memsize_conv3_batch, cudaMemcpyHostToDevice);

	// conv3_batch2
	float* dev_conv3_batch2_beta;
	cudaMalloc((void**)&dev_conv3_batch2_beta, memsize_conv3_batch);
	cudaMemcpy(dev_conv3_batch2_beta, conv3_batch2_beta, memsize_conv3_batch, cudaMemcpyHostToDevice);

	float* dev_conv3_batch2_gamma;
	cudaMalloc((void**)&dev_conv3_batch2_gamma, memsize_conv3_batch);
	cudaMemcpy(dev_conv3_batch2_gamma, conv3_batch2_gamma, memsize_conv3_batch, cudaMemcpyHostToDevice);

	float* dev_conv3_batch2_mean;
	cudaMalloc((void**)&dev_conv3_batch2_mean, memsize_conv3_batch);
	cudaMemcpy(dev_conv3_batch2_mean, conv3_batch2_mean, memsize_conv3_batch, cudaMemcpyHostToDevice);

	float* dev_conv3_batch2_std;
	cudaMalloc((void**)&dev_conv3_batch2_std, memsize_conv3_batch);
	cudaMemcpy(dev_conv3_batch2_std, conv3_batch2_std, memsize_conv3_batch, cudaMemcpyHostToDevice);

	// conv3_batch3
	float* dev_conv3_batch3_beta;
	cudaMalloc((void**)&dev_conv3_batch3_beta, memsize_conv3_batch);
	cudaMemcpy(dev_conv3_batch3_beta, conv3_batch3_beta, memsize_conv3_batch, cudaMemcpyHostToDevice);

	float* dev_conv3_batch3_gamma;
	cudaMalloc((void**)&dev_conv3_batch3_gamma, memsize_conv3_batch);
	cudaMemcpy(dev_conv3_batch3_gamma, conv3_batch3_gamma, memsize_conv3_batch, cudaMemcpyHostToDevice);

	float* dev_conv3_batch3_mean;
	cudaMalloc((void**)&dev_conv3_batch3_mean, memsize_conv3_batch);
	cudaMemcpy(dev_conv3_batch3_mean, conv3_batch3_mean, memsize_conv3_batch, cudaMemcpyHostToDevice);

	float* dev_conv3_batch3_std;
	cudaMalloc((void**)&dev_conv3_batch3_std, memsize_conv3_batch);
	cudaMemcpy(dev_conv3_batch3_std, conv3_batch3_std, memsize_conv3_batch, cudaMemcpyHostToDevice);

	// conv3_batch4
	float* dev_conv3_batch4_beta;
	cudaMalloc((void**)&dev_conv3_batch4_beta, memsize_conv3_batch);
	cudaMemcpy(dev_conv3_batch4_beta, conv3_batch4_beta, memsize_conv3_batch, cudaMemcpyHostToDevice);

	float* dev_conv3_batch4_gamma;
	cudaMalloc((void**)&dev_conv3_batch4_gamma, memsize_conv3_batch);
	cudaMemcpy(dev_conv3_batch4_gamma, conv3_batch4_gamma, memsize_conv3_batch, cudaMemcpyHostToDevice);

	float* dev_conv3_batch4_mean;
	cudaMalloc((void**)&dev_conv3_batch4_mean, memsize_conv3_batch);
	cudaMemcpy(dev_conv3_batch4_mean, conv3_batch4_mean, memsize_conv3_batch, cudaMemcpyHostToDevice);

	float* dev_conv3_batch4_std;
	cudaMalloc((void**)&dev_conv3_batch4_std, memsize_conv3_batch);
	cudaMemcpy(dev_conv3_batch4_std, conv3_batch4_std, memsize_conv3_batch, cudaMemcpyHostToDevice);

	// conv3_batch5
	float* dev_conv3_batch5_beta;
	cudaMalloc((void**)&dev_conv3_batch5_beta, memsize_conv3_batch);
	cudaMemcpy(dev_conv3_batch5_beta, conv3_batch5_beta, memsize_conv3_batch, cudaMemcpyHostToDevice);

	float* dev_conv3_batch5_gamma;
	cudaMalloc((void**)&dev_conv3_batch5_gamma, memsize_conv3_batch);
	cudaMemcpy(dev_conv3_batch5_gamma, conv3_batch5_gamma, memsize_conv3_batch, cudaMemcpyHostToDevice);

	float* dev_conv3_batch5_mean;
	cudaMalloc((void**)&dev_conv3_batch5_mean, memsize_conv3_batch);
	cudaMemcpy(dev_conv3_batch5_mean, conv3_batch5_mean, memsize_conv3_batch, cudaMemcpyHostToDevice);

	float* dev_conv3_batch5_std;
	cudaMalloc((void**)&dev_conv3_batch5_std, memsize_conv3_batch);
	cudaMemcpy(dev_conv3_batch5_std, conv3_batch5_std, memsize_conv3_batch, cudaMemcpyHostToDevice);

	// Device memory for intermediate tensors
	memsize_output_data = num_filters3 * (image_size/2) * (image_size/2) * sizeof(float);
	float* dev_layer3_output_data1;
	cudaMalloc((void**)&dev_layer3_output_data1, memsize_output_data);

	float* dev_layer3_output_data2;
	cudaMalloc((void**)&dev_layer3_output_data2, memsize_output_data);

	// Intermediate tensors for residual connections
	float* dev_layer3_output_data3;
	cudaMalloc((void**)&dev_layer3_output_data3, memsize_output_data);
	cudaMemset(&dev_layer3_output_data3, 0, memsize_output_data);

	float* dev_layer3_output_data4;
	cudaMalloc((void**)&dev_layer3_output_data4, memsize_output_data);

	// Convolution block
	// Define CudaKernel settings
	threadInBlock.x = 8;
	threadInBlock.y = 8;
	threadInBlock.z = 16;
	numBlocks.x = (image_size + threadInBlock.x - 1) / threadInBlock.x;
	numBlocks.y = (image_size + threadInBlock.y - 1) / threadInBlock.y;
	numBlocks.z = num_filters3;
	memsize_shared_memory = threadInBlock.x * threadInBlock.y * threadInBlock.z * sizeof(float);

	// conv3_1
	convolution_parallel << <numBlocks, threadInBlock, memsize_shared_memory >> > (dev_layer2_output_data1, image_size, image_size, num_filters2, dev_conv3_1_weights, kernel_size, stride, dev_layer3_output_data1);

	// conv3_3 (residual connection)
	convolution_parallel << <numBlocks, threadInBlock, memsize_shared_memory >> > (dev_layer2_output_data1, image_size, image_size, num_filters2, dev_conv3_3_weights, 1, stride, dev_layer3_output_data3);

	// Define CudaKernel settings
	image_size = image_size / 2;
	threadInBlock.x = 16;
	threadInBlock.y = 16;
	threadInBlock.z = 1;
	numBlocks.x = (image_size + threadInBlock.x - 1) / threadInBlock.x;
	numBlocks.y = (image_size + threadInBlock.y - 1) / threadInBlock.y;
	numBlocks.z = num_filters3;

	// BatchNorm
	batch_normalization_parallel << <numBlocks, threadInBlock >> > (dev_layer3_output_data1, image_size, image_size, dev_conv3_batch1_beta, dev_conv3_batch1_gamma, dev_conv3_batch1_mean, dev_conv3_batch1_std, dev_layer3_output_data2);

	// ReLU
	relu_parallel << <numBlocks, threadInBlock >> > (dev_layer3_output_data2, image_size, image_size, dev_layer3_output_data1);

	// BatchNorm (residual_connection)
	batch_normalization_parallel << <numBlocks, threadInBlock >> > (dev_layer3_output_data3, image_size, image_size, dev_conv3_batch3_beta, dev_conv3_batch3_gamma, dev_conv3_batch3_mean, dev_conv3_batch3_std, dev_layer3_output_data4);

	// Define CudaKernel settings
	threadInBlock.x = 8;
	threadInBlock.y = 8;
	threadInBlock.z = 16;
	numBlocks.x = (image_size + threadInBlock.x - 1) / threadInBlock.x;
	numBlocks.y = (image_size + threadInBlock.y - 1) / threadInBlock.y;
	numBlocks.z = num_filters3;
	memsize_shared_memory = threadInBlock.x * threadInBlock.y * threadInBlock.z * sizeof(float);

	// conv3_2
	convolution_parallel << <numBlocks, threadInBlock, memsize_shared_memory >> > (dev_layer3_output_data1, image_size, image_size, num_filters3, dev_conv3_2_weights, kernel_size, 1, dev_layer3_output_data2);

	// Define CudaKernel settings
	threadInBlock.x = 16;
	threadInBlock.y = 16;
	threadInBlock.z = 1;
	numBlocks.x = (image_size + threadInBlock.x - 1) / threadInBlock.x;
	numBlocks.y = (image_size + threadInBlock.y - 1) / threadInBlock.y;
	numBlocks.z = num_filters3;

	// BatchNorm
	batch_normalization_parallel << <numBlocks, threadInBlock >> > (dev_layer3_output_data2, image_size, image_size, dev_conv3_batch2_beta, dev_conv3_batch2_gamma, dev_conv3_batch2_mean, dev_conv3_batch2_std, dev_layer3_output_data1);

	// ResidualConnection
	add_tensors_parallel << <numBlocks, threadInBlock >> > (dev_layer3_output_data1, dev_layer3_output_data4, image_size, image_size, dev_layer3_output_data2);

	// ReLU
	relu_parallel << <numBlocks, threadInBlock >> > (dev_layer3_output_data2, image_size, image_size, dev_layer3_output_data1);

	// Debug
	if (debug) {
		memsize_debug = image_size * image_size * num_filters3 * sizeof(float);
		debug_data = (float*)malloc(memsize_debug);
		cudaMemcpy(debug_data, dev_layer3_output_data1, memsize_debug, cudaMemcpyDeviceToHost);
		float max = FLT_MIN;
		float min = FLT_MAX;
		for (int i = 0; i < image_size * image_size * num_filters3; i++) {
			if (debug_data[i] > max) {
				max = debug_data[i];
			}
			if (debug_data[i] < min) {
				min = debug_data[i];
			}
		}
		printf("Output layer 3 (Convolution block):\n");
		printf("Max value: %f\n", max);
		printf("Min value: %f\n", min);
	}

	// Identity block
	// ReLU (residual connection)
	relu_parallel << <numBlocks, threadInBlock >> > (dev_layer3_output_data1, image_size, image_size, dev_layer3_output_data3);

	// Define CudaKernel settings
	threadInBlock.x = 8;
	threadInBlock.y = 8;
	threadInBlock.z = 16;
	numBlocks.x = (image_size + threadInBlock.x - 1) / threadInBlock.x;
	numBlocks.y = (image_size + threadInBlock.y - 1) / threadInBlock.y;
	numBlocks.z = num_filters3;
	memsize_shared_memory = threadInBlock.x * threadInBlock.y * threadInBlock.z * sizeof(float);

	// conv3_4
	convolution_parallel << <numBlocks, threadInBlock, memsize_shared_memory >> > (dev_layer3_output_data1, image_size, image_size, num_filters3, dev_conv3_4_weights, kernel_size, 1, dev_layer3_output_data2);

	// Define CudaKernel settings
	threadInBlock.x = 16;
	threadInBlock.y = 16;
	threadInBlock.z = 1;
	numBlocks.x = (image_size + threadInBlock.x - 1) / threadInBlock.x;
	numBlocks.y = (image_size + threadInBlock.y - 1) / threadInBlock.y;
	numBlocks.z = num_filters3;

	// BatchNorm
	batch_normalization_parallel << <numBlocks, threadInBlock >> > (dev_layer3_output_data2, image_size, image_size, dev_conv3_batch4_beta, dev_conv3_batch4_gamma, dev_conv3_batch4_mean, dev_conv3_batch4_std, dev_layer3_output_data1);

	// ReLU
	relu_parallel << <numBlocks, threadInBlock >> > (dev_layer3_output_data1, image_size, image_size, dev_layer3_output_data2);

	// Define CudaKernel settings
	threadInBlock.x = 8;
	threadInBlock.y = 8;
	threadInBlock.z = 16;
	numBlocks.x = (image_size + threadInBlock.x - 1) / threadInBlock.x;
	numBlocks.y = (image_size + threadInBlock.y - 1) / threadInBlock.y;
	numBlocks.z = num_filters3;
	memsize_shared_memory = threadInBlock.x * threadInBlock.y * threadInBlock.z * sizeof(float);

	// conv3_5
	convolution_parallel << <numBlocks, threadInBlock, memsize_shared_memory >> > (dev_layer3_output_data2, image_size, image_size, num_filters3, dev_conv3_5_weights, kernel_size, 1, dev_layer3_output_data1);

	// Define CudaKernel settings
	threadInBlock.x = 16;
	threadInBlock.y = 16;
	threadInBlock.z = 1;
	numBlocks.x = (image_size + threadInBlock.x - 1) / threadInBlock.x;
	numBlocks.y = (image_size + threadInBlock.y - 1) / threadInBlock.y;
	numBlocks.z = num_filters3;

	// BatchNorm
	batch_normalization_parallel << <numBlocks, threadInBlock >> > (dev_layer3_output_data1, image_size, image_size, dev_conv3_batch5_beta, dev_conv3_batch5_gamma, dev_conv3_batch5_mean, dev_conv3_batch5_std, dev_layer3_output_data2);

	// ResidualConnection
	add_tensors_parallel << <numBlocks, threadInBlock >> > (dev_layer3_output_data2, dev_layer3_output_data3, image_size, image_size, dev_layer3_output_data1);

	// ReLU
	relu_parallel << <numBlocks, threadInBlock >> > (dev_layer3_output_data1, image_size, image_size, dev_layer3_output_data2);

	// Debug
	if (debug) {
		memsize_debug = image_size * image_size * num_filters3 * sizeof(float);
		debug_data = (float*)malloc(memsize_debug);
		cudaMemcpy(debug_data, dev_layer3_output_data2, memsize_debug, cudaMemcpyDeviceToHost);
		float max = FLT_MIN;
		float min = FLT_MAX;
		for (int i = 0; i < image_size * image_size * num_filters3; i++) {
			if (debug_data[i] > max) {
				max = debug_data[i];
			}
			if (debug_data[i] < min) {
				min = debug_data[i];
			}
		}
		printf("Output layer 3 (Identity block):\n");
		printf("Max value: %f\n", max);
		printf("Min value: %f\n", min);
	}

	//=================================
	//							 Layer 4 
	//=================================
	// Allocate device (GPU) memory for kernels and intermediate tensors
	// Device memory for conv3_x_weights
	float* dev_conv4_1_weights;
	cudaMalloc((void**)&dev_conv4_1_weights, memsize_conv4_1_weights);

	float* dev_conv4_2_weights;
	cudaMalloc((void**)&dev_conv4_2_weights, memsize_conv4_2_weights);

	float* dev_conv4_3_weights;
	cudaMalloc((void**)&dev_conv4_3_weights, memsize_conv4_3_weights);

	float* dev_conv4_4_weights;
	cudaMalloc((void**)&dev_conv4_4_weights, memsize_conv4_4_weights);

	float* dev_conv4_5_weights;
	cudaMalloc((void**)&dev_conv4_5_weights, memsize_conv4_5_weights);

	for (int i = 0; i < num_filters4; i++) {
		cudaMemcpy(dev_conv4_1_weights + i * kernel_size * kernel_size * num_filters3, conv4_1_weights[i].data, kernel_size * kernel_size * num_filters3 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_conv4_2_weights + i * kernel_size * kernel_size * num_filters4, conv4_2_weights[i].data, kernel_size * kernel_size * num_filters4 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_conv4_3_weights + i * 1 * 1 * num_filters3, conv4_3_weights[i].data, 1 * 1 * num_filters3 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_conv4_4_weights + i * kernel_size * kernel_size * num_filters4, conv4_4_weights[i].data, kernel_size * kernel_size * num_filters4 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_conv4_5_weights + i * kernel_size * kernel_size * num_filters4, conv4_5_weights[i].data, kernel_size * kernel_size * num_filters4 * sizeof(float), cudaMemcpyHostToDevice);
	}

	// Device memory for BatchNorm
	int memsize_conv4_batch = num_filters4 * sizeof(float);

	// conv4_batch1
	float* dev_conv4_batch1_beta;
	cudaMalloc((void**)&dev_conv4_batch1_beta, memsize_conv4_batch);
	cudaMemcpy(dev_conv4_batch1_beta, conv4_batch1_beta, memsize_conv4_batch, cudaMemcpyHostToDevice);

	float* dev_conv4_batch1_gamma;
	cudaMalloc((void**)&dev_conv4_batch1_gamma, memsize_conv4_batch);
	cudaMemcpy(dev_conv4_batch1_gamma, conv4_batch1_gamma, memsize_conv4_batch, cudaMemcpyHostToDevice);

	float* dev_conv4_batch1_mean;
	cudaMalloc((void**)&dev_conv4_batch1_mean, memsize_conv4_batch);
	cudaMemcpy(dev_conv4_batch1_mean, conv4_batch1_mean, memsize_conv4_batch, cudaMemcpyHostToDevice);

	float* dev_conv4_batch1_std;
	cudaMalloc((void**)&dev_conv4_batch1_std, memsize_conv4_batch);
	cudaMemcpy(dev_conv4_batch1_std, conv4_batch1_std, memsize_conv4_batch, cudaMemcpyHostToDevice);

	// conv4_batch2
	float* dev_conv4_batch2_beta;
	cudaMalloc((void**)&dev_conv4_batch2_beta, memsize_conv4_batch);
	cudaMemcpy(dev_conv4_batch2_beta, conv4_batch2_beta, memsize_conv4_batch, cudaMemcpyHostToDevice);

	float* dev_conv4_batch2_gamma;
	cudaMalloc((void**)&dev_conv4_batch2_gamma, memsize_conv4_batch);
	cudaMemcpy(dev_conv4_batch2_gamma, conv4_batch2_gamma, memsize_conv4_batch, cudaMemcpyHostToDevice);

	float* dev_conv4_batch2_mean;
	cudaMalloc((void**)&dev_conv4_batch2_mean, memsize_conv4_batch);
	cudaMemcpy(dev_conv4_batch2_mean, conv4_batch2_mean, memsize_conv4_batch, cudaMemcpyHostToDevice);

	float* dev_conv4_batch2_std;
	cudaMalloc((void**)&dev_conv4_batch2_std, memsize_conv4_batch);
	cudaMemcpy(dev_conv4_batch2_std, conv4_batch2_std, memsize_conv4_batch, cudaMemcpyHostToDevice);

	// conv4_batch3
	float* dev_conv4_batch3_beta;
	cudaMalloc((void**)&dev_conv4_batch3_beta, memsize_conv4_batch);
	cudaMemcpy(dev_conv4_batch3_beta, conv4_batch3_beta, memsize_conv4_batch, cudaMemcpyHostToDevice);

	float* dev_conv4_batch3_gamma;
	cudaMalloc((void**)&dev_conv4_batch3_gamma, memsize_conv4_batch);
	cudaMemcpy(dev_conv4_batch3_gamma, conv4_batch3_gamma, memsize_conv4_batch, cudaMemcpyHostToDevice);

	float* dev_conv4_batch3_mean;
	cudaMalloc((void**)&dev_conv4_batch3_mean, memsize_conv4_batch);
	cudaMemcpy(dev_conv4_batch3_mean, conv4_batch3_mean, memsize_conv4_batch, cudaMemcpyHostToDevice);

	float* dev_conv4_batch3_std;
	cudaMalloc((void**)&dev_conv4_batch3_std, memsize_conv4_batch);
	cudaMemcpy(dev_conv4_batch3_std, conv4_batch3_std, memsize_conv4_batch, cudaMemcpyHostToDevice);

	// conv4_batch4
	float* dev_conv4_batch4_beta;
	cudaMalloc((void**)&dev_conv4_batch4_beta, memsize_conv4_batch);
	cudaMemcpy(dev_conv4_batch4_beta, conv4_batch4_beta, memsize_conv4_batch, cudaMemcpyHostToDevice);

	float* dev_conv4_batch4_gamma;
	cudaMalloc((void**)&dev_conv4_batch4_gamma, memsize_conv4_batch);
	cudaMemcpy(dev_conv4_batch4_gamma, conv4_batch4_gamma, memsize_conv4_batch, cudaMemcpyHostToDevice);

	float* dev_conv4_batch4_mean;
	cudaMalloc((void**)&dev_conv4_batch4_mean, memsize_conv4_batch);
	cudaMemcpy(dev_conv4_batch4_mean, conv4_batch4_mean, memsize_conv4_batch, cudaMemcpyHostToDevice);

	float* dev_conv4_batch4_std;
	cudaMalloc((void**)&dev_conv4_batch4_std, memsize_conv4_batch);
	cudaMemcpy(dev_conv4_batch4_std, conv4_batch4_std, memsize_conv4_batch, cudaMemcpyHostToDevice);

	// conv4_batch5
	float* dev_conv4_batch5_beta;
	cudaMalloc((void**)&dev_conv4_batch5_beta, memsize_conv4_batch);
	cudaMemcpy(dev_conv4_batch5_beta, conv4_batch5_beta, memsize_conv4_batch, cudaMemcpyHostToDevice);

	float* dev_conv4_batch5_gamma;
	cudaMalloc((void**)&dev_conv4_batch5_gamma, memsize_conv4_batch);
	cudaMemcpy(dev_conv4_batch5_gamma, conv4_batch5_gamma, memsize_conv4_batch, cudaMemcpyHostToDevice);

	float* dev_conv4_batch5_mean;
	cudaMalloc((void**)&dev_conv4_batch5_mean, memsize_conv4_batch);
	cudaMemcpy(dev_conv4_batch5_mean, conv4_batch5_mean, memsize_conv4_batch, cudaMemcpyHostToDevice);

	float* dev_conv4_batch5_std;
	cudaMalloc((void**)&dev_conv4_batch5_std, memsize_conv4_batch);
	cudaMemcpy(dev_conv4_batch5_std, conv4_batch5_std, memsize_conv4_batch, cudaMemcpyHostToDevice);

	// Device memory for intermediate tensors
	memsize_output_data = num_filters4 * (image_size / 2) * (image_size / 2) * sizeof(float);
	float* dev_layer4_output_data1;
	cudaMalloc((void**)&dev_layer4_output_data1, memsize_output_data);

	float* dev_layer4_output_data2;
	cudaMalloc((void**)&dev_layer4_output_data2, memsize_output_data);

	// Intermediate tensors for residual connections
	float* dev_layer4_output_data3;
	cudaMalloc((void**)&dev_layer4_output_data3, memsize_output_data);

	float* dev_layer4_output_data4;
	cudaMalloc((void**)&dev_layer4_output_data4, memsize_output_data);

	// Convolution block
	// Define CudaKernel settings
	threadInBlock.x = 8;
	threadInBlock.y = 8;
	threadInBlock.z = 16;
	numBlocks.x = (image_size + threadInBlock.x - 1) / threadInBlock.x;
	numBlocks.y = (image_size + threadInBlock.y - 1) / threadInBlock.y;
	numBlocks.z = num_filters4;
	memsize_shared_memory = threadInBlock.x * threadInBlock.y * threadInBlock.z * sizeof(float);

	// conv4_1
	convolution_parallel << <numBlocks, threadInBlock, memsize_shared_memory >> > (dev_layer3_output_data2, image_size, image_size, num_filters3, dev_conv4_1_weights, kernel_size, stride, dev_layer4_output_data1);

	// conv4_3 (residual connection)
	convolution_parallel << <numBlocks, threadInBlock, memsize_shared_memory >> > (dev_layer3_output_data2, image_size, image_size, num_filters3, dev_conv4_3_weights, 1, stride, dev_layer4_output_data3);

	// Define CudaKernel settings
	image_size = image_size / 2;
	threadInBlock.x = 16;
	threadInBlock.y = 16;
	threadInBlock.z = 1;
	numBlocks.x = (image_size + threadInBlock.x - 1) / threadInBlock.x;
	numBlocks.y = (image_size + threadInBlock.y - 1) / threadInBlock.y;
	numBlocks.z = num_filters4;

	// BatchNorm
	batch_normalization_parallel << <numBlocks, threadInBlock >> > (dev_layer4_output_data1, image_size, image_size, dev_conv4_batch1_beta, dev_conv4_batch1_gamma, dev_conv4_batch1_mean, dev_conv4_batch1_std, dev_layer4_output_data2);

	// ReLU
	relu_parallel << <numBlocks, threadInBlock >> > (dev_layer4_output_data2, image_size, image_size, dev_layer4_output_data1);

	// BatchNorm (residual_connection)
	batch_normalization_parallel << <numBlocks, threadInBlock >> > (dev_layer4_output_data3, image_size, image_size, dev_conv4_batch3_beta, dev_conv4_batch3_gamma, dev_conv4_batch3_mean, dev_conv4_batch3_std, dev_layer4_output_data4);

	// Define CudaKernel settings
	threadInBlock.x = 8;
	threadInBlock.y = 8;
	threadInBlock.z = 16;
	numBlocks.x = (image_size + threadInBlock.x - 1) / threadInBlock.x;
	numBlocks.y = (image_size + threadInBlock.y - 1) / threadInBlock.y;
	numBlocks.z = num_filters4;
	memsize_shared_memory = threadInBlock.x * threadInBlock.y * threadInBlock.z * sizeof(float);

	// conv4_2
	convolution_parallel << <numBlocks, threadInBlock, memsize_shared_memory >> > (dev_layer4_output_data1, image_size, image_size, num_filters4, dev_conv4_2_weights, kernel_size, 1, dev_layer4_output_data2);

	// Define CudaKernel settings
	threadInBlock.x = 16;
	threadInBlock.y = 16;
	threadInBlock.z = 1;
	numBlocks.x = (image_size + threadInBlock.x - 1) / threadInBlock.x;
	numBlocks.y = (image_size + threadInBlock.y - 1) / threadInBlock.y;
	numBlocks.z = num_filters4;

	// BatchNorm
	batch_normalization_parallel << <numBlocks, threadInBlock >> > (dev_layer4_output_data2, image_size, image_size, dev_conv4_batch2_beta, dev_conv4_batch2_gamma, dev_conv4_batch2_mean, dev_conv4_batch2_std, dev_layer4_output_data1);

	// ResidualConnection
	add_tensors_parallel << <numBlocks, threadInBlock >> > (dev_layer4_output_data1, dev_layer4_output_data4, image_size, image_size, dev_layer4_output_data2);

	// ReLU
	relu_parallel << <numBlocks, threadInBlock >> > (dev_layer4_output_data2, image_size, image_size, dev_layer4_output_data1);

	// Debug
	if (debug) {
		memsize_debug = image_size * image_size * num_filters4 * sizeof(float);
		debug_data = (float*)malloc(memsize_debug);
		cudaMemcpy(debug_data, dev_layer4_output_data1, memsize_debug, cudaMemcpyDeviceToHost);
		float max = FLT_MIN;
		float min = FLT_MAX;
		for (int i = 0; i < image_size * image_size * num_filters4; i++) {
			if (debug_data[i] > max) {
				max = debug_data[i];
			}
			if (debug_data[i] < min) {
				min = debug_data[i];
			}
		}
		printf("Output layer 4 (Convolution block):\n");
		printf("Max value: %f\n", max);
		printf("Min value: %f\n", min);
	}

	// Identity block
	// ReLU (residual connection)
	relu_parallel << <numBlocks, threadInBlock >> > (dev_layer4_output_data1, image_size, image_size, dev_layer4_output_data3);

	// Define CudaKernel settings
	threadInBlock.x = 8;
	threadInBlock.y = 8;
	threadInBlock.z = 16;
	numBlocks.x = (image_size + threadInBlock.x - 1) / threadInBlock.x;
	numBlocks.y = (image_size + threadInBlock.y - 1) / threadInBlock.y;
	numBlocks.z = num_filters4;
	memsize_shared_memory = threadInBlock.x * threadInBlock.y * threadInBlock.z * sizeof(float);

	// conv4_4
	convolution_parallel << <numBlocks, threadInBlock, memsize_shared_memory >> > (dev_layer4_output_data1, image_size, image_size, num_filters4, dev_conv4_4_weights, kernel_size, 1, dev_layer4_output_data2);

	// Define CudaKernel settings
	threadInBlock.x = 16;
	threadInBlock.y = 16;
	threadInBlock.z = 1;
	numBlocks.x = (image_size + threadInBlock.x - 1) / threadInBlock.x;
	numBlocks.y = (image_size + threadInBlock.y - 1) / threadInBlock.y;
	numBlocks.z = num_filters4;

	// BatchNorm
	batch_normalization_parallel << <numBlocks, threadInBlock >> > (dev_layer4_output_data2, image_size, image_size, dev_conv4_batch4_beta, dev_conv4_batch4_gamma, dev_conv4_batch4_mean, dev_conv4_batch4_std, dev_layer4_output_data1);

	// ReLU
	relu_parallel << <numBlocks, threadInBlock >> > (dev_layer4_output_data1, image_size, image_size, dev_layer4_output_data2);

	// Define CudaKernel settings
	threadInBlock.x = 8;
	threadInBlock.y = 8;
	threadInBlock.z = 16;
	numBlocks.x = (image_size + threadInBlock.x - 1) / threadInBlock.x;
	numBlocks.y = (image_size + threadInBlock.y - 1) / threadInBlock.y;
	numBlocks.z = num_filters4;
	memsize_shared_memory = threadInBlock.x * threadInBlock.y * threadInBlock.z * sizeof(float);

	// conv4_5
	convolution_parallel << <numBlocks, threadInBlock, memsize_shared_memory >> > (dev_layer4_output_data2, image_size, image_size, num_filters4, dev_conv4_5_weights, kernel_size, 1, dev_layer4_output_data1);

	// Define CudaKernel settings
	threadInBlock.x = 16;
	threadInBlock.y = 16;
	threadInBlock.z = 1;
	numBlocks.x = (image_size + threadInBlock.x - 1) / threadInBlock.x;
	numBlocks.y = (image_size + threadInBlock.y - 1) / threadInBlock.y;
	numBlocks.z = num_filters4;

	// BatchNorm
	batch_normalization_parallel << <numBlocks, threadInBlock >> > (dev_layer4_output_data1, image_size, image_size, dev_conv4_batch5_beta, dev_conv4_batch5_gamma, dev_conv4_batch5_mean, dev_conv4_batch5_std, dev_layer4_output_data2);

	// ResidualConnection
	add_tensors_parallel << <numBlocks, threadInBlock >> > (dev_layer4_output_data2, dev_layer4_output_data3, image_size, image_size, dev_layer4_output_data1);

	// ReLU
	relu_parallel << <numBlocks, threadInBlock >> > (dev_layer4_output_data1, image_size, image_size, dev_layer4_output_data2);

	// Debug
	if (debug) {
		memsize_debug = image_size * image_size * num_filters4 * sizeof(float);
		debug_data = (float*)malloc(memsize_debug);
		cudaMemcpy(debug_data, dev_layer4_output_data2, memsize_debug, cudaMemcpyDeviceToHost);
		float max = FLT_MIN;
		float min = FLT_MAX;
		for (int i = 0; i < image_size * image_size * num_filters4; i++) {
			if (debug_data[i] > max) {
				max = debug_data[i];
			}
			if (debug_data[i] < min) {
				min = debug_data[i];
			}
		}
		printf("Output layer 4 (Identity block):\n");
		printf("Max value: %f\n", max);
		printf("Min value: %f\n", min);
	}

	//=================================
	//							 Layer 5 
	//=================================
	// Allocate device (GPU) memory for kernels and intermediate tensors
	// Device memory for conv5_x_weights
	float* dev_conv5_1_weights;
	cudaMalloc((void**)&dev_conv5_1_weights, memsize_conv5_1_weights);

	float* dev_conv5_2_weights;
	cudaMalloc((void**)&dev_conv5_2_weights, memsize_conv5_2_weights);

	float* dev_conv5_3_weights;
	cudaMalloc((void**)&dev_conv5_3_weights, memsize_conv5_3_weights);

	float* dev_conv5_4_weights;
	cudaMalloc((void**)&dev_conv5_4_weights, memsize_conv5_4_weights);

	float* dev_conv5_5_weights;
	cudaMalloc((void**)&dev_conv5_5_weights, memsize_conv5_5_weights);

	for (int i = 0; i < num_filters5; i++) {
		cudaMemcpy(dev_conv5_1_weights + i * kernel_size * kernel_size * num_filters4, conv5_1_weights[i].data, kernel_size * kernel_size * num_filters4 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_conv5_2_weights + i * kernel_size * kernel_size * num_filters5, conv5_2_weights[i].data, kernel_size * kernel_size * num_filters5 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_conv5_3_weights + i * 1 * 1 * num_filters4, conv5_3_weights[i].data, 1 * 1 * num_filters4 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_conv5_4_weights + i * kernel_size * kernel_size * num_filters5, conv5_4_weights[i].data, kernel_size * kernel_size * num_filters5 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_conv5_5_weights + i * kernel_size * kernel_size * num_filters5, conv5_5_weights[i].data, kernel_size * kernel_size * num_filters5 * sizeof(float), cudaMemcpyHostToDevice);
	}

	// Device memory for BatchNorm
	int memsize_conv5_batch = num_filters5 * sizeof(float);

	// conv5_batch1
	float* dev_conv5_batch1_beta;
	cudaMalloc((void**)&dev_conv5_batch1_beta, memsize_conv5_batch);
	cudaMemcpy(dev_conv5_batch1_beta, conv5_batch1_beta, memsize_conv5_batch, cudaMemcpyHostToDevice);

	float* dev_conv5_batch1_gamma;
	cudaMalloc((void**)&dev_conv5_batch1_gamma, memsize_conv5_batch);
	cudaMemcpy(dev_conv5_batch1_gamma, conv5_batch1_gamma, memsize_conv5_batch, cudaMemcpyHostToDevice);

	float* dev_conv5_batch1_mean;
	cudaMalloc((void**)&dev_conv5_batch1_mean, memsize_conv5_batch);
	cudaMemcpy(dev_conv5_batch1_mean, conv5_batch1_mean, memsize_conv5_batch, cudaMemcpyHostToDevice);

	float* dev_conv5_batch1_std;
	cudaMalloc((void**)&dev_conv5_batch1_std, memsize_conv5_batch);
	cudaMemcpy(dev_conv5_batch1_std, conv5_batch1_std, memsize_conv5_batch, cudaMemcpyHostToDevice);

	// conv5_batch2
	float* dev_conv5_batch2_beta;
	cudaMalloc((void**)&dev_conv5_batch2_beta, memsize_conv5_batch);
	cudaMemcpy(dev_conv5_batch2_beta, conv5_batch2_beta, memsize_conv5_batch, cudaMemcpyHostToDevice);

	float* dev_conv5_batch2_gamma;
	cudaMalloc((void**)&dev_conv5_batch2_gamma, memsize_conv5_batch);
	cudaMemcpy(dev_conv5_batch2_gamma, conv5_batch2_gamma, memsize_conv5_batch, cudaMemcpyHostToDevice);

	float* dev_conv5_batch2_mean;
	cudaMalloc((void**)&dev_conv5_batch2_mean, memsize_conv5_batch);
	cudaMemcpy(dev_conv5_batch2_mean, conv5_batch2_mean, memsize_conv5_batch, cudaMemcpyHostToDevice);

	float* dev_conv5_batch2_std;
	cudaMalloc((void**)&dev_conv5_batch2_std, memsize_conv5_batch);
	cudaMemcpy(dev_conv5_batch2_std, conv5_batch2_std, memsize_conv5_batch, cudaMemcpyHostToDevice);

	// conv5_batch3
	float* dev_conv5_batch3_beta;
	cudaMalloc((void**)&dev_conv5_batch3_beta, memsize_conv5_batch);
	cudaMemcpy(dev_conv5_batch3_beta, conv5_batch3_beta, memsize_conv5_batch, cudaMemcpyHostToDevice);

	float* dev_conv5_batch3_gamma;
	cudaMalloc((void**)&dev_conv5_batch3_gamma, memsize_conv5_batch);
	cudaMemcpy(dev_conv5_batch3_gamma, conv5_batch3_gamma, memsize_conv5_batch, cudaMemcpyHostToDevice);

	float* dev_conv5_batch3_mean;
	cudaMalloc((void**)&dev_conv5_batch3_mean, memsize_conv5_batch);
	cudaMemcpy(dev_conv5_batch3_mean, conv5_batch3_mean, memsize_conv5_batch, cudaMemcpyHostToDevice);

	float* dev_conv5_batch3_std;
	cudaMalloc((void**)&dev_conv5_batch3_std, memsize_conv5_batch);
	cudaMemcpy(dev_conv5_batch3_std, conv5_batch3_std, memsize_conv5_batch, cudaMemcpyHostToDevice);

	// conv5_batch4
	float* dev_conv5_batch4_beta;
	cudaMalloc((void**)&dev_conv5_batch4_beta, memsize_conv5_batch);
	cudaMemcpy(dev_conv5_batch4_beta, conv5_batch4_beta, memsize_conv5_batch, cudaMemcpyHostToDevice);

	float* dev_conv5_batch4_gamma;
	cudaMalloc((void**)&dev_conv5_batch4_gamma, memsize_conv5_batch);
	cudaMemcpy(dev_conv5_batch4_gamma, conv5_batch4_gamma, memsize_conv5_batch, cudaMemcpyHostToDevice);

	float* dev_conv5_batch4_mean;
	cudaMalloc((void**)&dev_conv5_batch4_mean, memsize_conv5_batch);
	cudaMemcpy(dev_conv5_batch4_mean, conv5_batch4_mean, memsize_conv5_batch, cudaMemcpyHostToDevice);

	float* dev_conv5_batch4_std;
	cudaMalloc((void**)&dev_conv5_batch4_std, memsize_conv5_batch);
	cudaMemcpy(dev_conv5_batch4_std, conv5_batch4_std, memsize_conv5_batch, cudaMemcpyHostToDevice);

	// conv5_batch5
	float* dev_conv5_batch5_beta;
	cudaMalloc((void**)&dev_conv5_batch5_beta, memsize_conv5_batch);
	cudaMemcpy(dev_conv5_batch5_beta, conv5_batch5_beta, memsize_conv5_batch, cudaMemcpyHostToDevice);

	float* dev_conv5_batch5_gamma;
	cudaMalloc((void**)&dev_conv5_batch5_gamma, memsize_conv5_batch);
	cudaMemcpy(dev_conv5_batch5_gamma, conv5_batch5_gamma, memsize_conv5_batch, cudaMemcpyHostToDevice);

	float* dev_conv5_batch5_mean;
	cudaMalloc((void**)&dev_conv5_batch5_mean, memsize_conv5_batch);
	cudaMemcpy(dev_conv5_batch5_mean, conv5_batch5_mean, memsize_conv5_batch, cudaMemcpyHostToDevice);

	float* dev_conv5_batch5_std;
	cudaMalloc((void**)&dev_conv5_batch5_std, memsize_conv5_batch);
	cudaMemcpy(dev_conv5_batch5_std, conv5_batch5_std, memsize_conv5_batch, cudaMemcpyHostToDevice);

	// Device memory for intermediate tensors
	memsize_output_data = num_filters5 * (image_size / 2) * (image_size / 2) * sizeof(float);
	float* dev_layer5_output_data1;
	cudaMalloc((void**)&dev_layer5_output_data1, memsize_output_data);

	float* dev_layer5_output_data2;
	cudaMalloc((void**)&dev_layer5_output_data2, memsize_output_data);

	// Intermediate tensors for residual connections
	float* dev_layer5_output_data3;
	cudaMalloc((void**)&dev_layer5_output_data3, memsize_output_data);

	float* dev_layer5_output_data4;
	cudaMalloc((void**)&dev_layer5_output_data4, memsize_output_data);

	// Convolution block
	// Define CudaKernel settings
	threadInBlock.x = 8;
	threadInBlock.y = 8;
	threadInBlock.z = 16;
	numBlocks.x = (image_size + threadInBlock.x - 1) / threadInBlock.x;
	numBlocks.y = (image_size + threadInBlock.y - 1) / threadInBlock.y;
	numBlocks.z = num_filters5;
	memsize_shared_memory = threadInBlock.x * threadInBlock.y * threadInBlock.z * sizeof(float);

	// conv5_1
	convolution_parallel << <numBlocks, threadInBlock, memsize_shared_memory >> > (dev_layer4_output_data2, image_size, image_size, num_filters4, dev_conv5_1_weights, kernel_size, stride, dev_layer5_output_data1);

	// conv5_3 (residual connection)
	convolution_parallel << <numBlocks, threadInBlock, memsize_shared_memory >> > (dev_layer4_output_data2, image_size, image_size, num_filters4, dev_conv5_3_weights, 1, stride, dev_layer5_output_data3);

	// Define CudaKernel settings
	image_size = image_size / 2;
	threadInBlock.x = 16;
	threadInBlock.y = 16;
	threadInBlock.z = 1;
	numBlocks.x = (image_size + threadInBlock.x - 1) / threadInBlock.x;
	numBlocks.y = (image_size + threadInBlock.y - 1) / threadInBlock.y;
	numBlocks.z = num_filters5;

	// BatchNorm
	batch_normalization_parallel << <numBlocks, threadInBlock >> > (dev_layer5_output_data1, image_size, image_size, dev_conv5_batch1_beta, dev_conv5_batch1_gamma, dev_conv5_batch1_mean, dev_conv5_batch1_std, dev_layer5_output_data2);

	// ReLU
	relu_parallel << <numBlocks, threadInBlock >> > (dev_layer5_output_data2, image_size, image_size, dev_layer5_output_data1);

	// BatchNorm (residual_connection)
	batch_normalization_parallel << <numBlocks, threadInBlock >> > (dev_layer5_output_data3, image_size, image_size, dev_conv5_batch3_beta, dev_conv5_batch3_gamma, dev_conv5_batch3_mean, dev_conv5_batch3_std, dev_layer5_output_data4);

	// Define CudaKernel settings
	threadInBlock.x = 8;
	threadInBlock.y = 8;
	threadInBlock.z = 16;
	numBlocks.x = (image_size + threadInBlock.x - 1) / threadInBlock.x;
	numBlocks.y = (image_size + threadInBlock.y - 1) / threadInBlock.y;
	numBlocks.z = num_filters5;
	memsize_shared_memory = threadInBlock.x * threadInBlock.y * threadInBlock.z * sizeof(float);

	// conv5_2
	convolution_parallel << <numBlocks, threadInBlock, memsize_shared_memory >> > (dev_layer5_output_data1, image_size, image_size, num_filters5, dev_conv5_2_weights, kernel_size, 1, dev_layer5_output_data2);

	// Define CudaKernel settings
	threadInBlock.x = 16;
	threadInBlock.y = 16;
	threadInBlock.z = 1;
	numBlocks.x = (image_size + threadInBlock.x - 1) / threadInBlock.x;
	numBlocks.y = (image_size + threadInBlock.y - 1) / threadInBlock.y;
	numBlocks.z = num_filters5;

	// BatchNorm
	batch_normalization_parallel << <numBlocks, threadInBlock >> > (dev_layer5_output_data2, image_size, image_size, dev_conv5_batch2_beta, dev_conv5_batch2_gamma, dev_conv5_batch2_mean, dev_conv5_batch2_std, dev_layer5_output_data1);

	// ResidualConnection
	add_tensors_parallel << <numBlocks, threadInBlock >> > (dev_layer5_output_data1, dev_layer5_output_data4, image_size, image_size, dev_layer5_output_data2);

	// ReLU
	relu_parallel << <numBlocks, threadInBlock >> > (dev_layer5_output_data2, image_size, image_size, dev_layer5_output_data1);

	// Debug
	if (debug) {
		memsize_debug = image_size * image_size * num_filters5 * sizeof(float);
		debug_data = (float*)malloc(memsize_debug);
		cudaMemcpy(debug_data, dev_layer5_output_data1, memsize_debug, cudaMemcpyDeviceToHost);
		float max = FLT_MIN;
		float min = FLT_MAX;
		for (int i = 0; i < image_size * image_size * num_filters5; i++) {
			if (debug_data[i] > max) {
				max = debug_data[i];
			}
			if (debug_data[i] < min) {
				min = debug_data[i];
			}
		}
		printf("Output layer 5 (Convolution block):\n");
		printf("Max value: %f\n", max);
		printf("Min value: %f\n", min);
	}

	// Identity block
	// ReLU (residual connection)
	relu_parallel << <numBlocks, threadInBlock >> > (dev_layer5_output_data1, image_size, image_size, dev_layer5_output_data3);

	// Define CudaKernel settings
	threadInBlock.x = 8;
	threadInBlock.y = 8;
	threadInBlock.z = 16;
	numBlocks.x = (image_size + threadInBlock.x - 1) / threadInBlock.x;
	numBlocks.y = (image_size + threadInBlock.y - 1) / threadInBlock.y;
	numBlocks.z = num_filters5;
	memsize_shared_memory = threadInBlock.x * threadInBlock.y * threadInBlock.z * sizeof(float);

	// conv5_4
	convolution_parallel << <numBlocks, threadInBlock, memsize_shared_memory >> > (dev_layer5_output_data1, image_size, image_size, num_filters5, dev_conv5_4_weights, kernel_size, 1, dev_layer5_output_data2);

	// Define CudaKernel settings
	threadInBlock.x = 16;
	threadInBlock.y = 16;
	threadInBlock.z = 1;
	numBlocks.x = (image_size + threadInBlock.x - 1) / threadInBlock.x;
	numBlocks.y = (image_size + threadInBlock.y - 1) / threadInBlock.y;
	numBlocks.z = num_filters5;

	// BatchNorm
	batch_normalization_parallel << <numBlocks, threadInBlock >> > (dev_layer5_output_data2, image_size, image_size, dev_conv5_batch4_beta, dev_conv5_batch4_gamma, dev_conv5_batch4_mean, dev_conv5_batch4_std, dev_layer5_output_data1);

	// ReLU
	relu_parallel << <numBlocks, threadInBlock >> > (dev_layer5_output_data1, image_size, image_size, dev_layer5_output_data2);

	// Define CudaKernel settings
	threadInBlock.x = 8;
	threadInBlock.y = 8;
	threadInBlock.z = 16;
	numBlocks.x = (image_size + threadInBlock.x - 1) / threadInBlock.x;
	numBlocks.y = (image_size + threadInBlock.y - 1) / threadInBlock.y;
	numBlocks.z = num_filters5;
	memsize_shared_memory = threadInBlock.x * threadInBlock.y * threadInBlock.z * sizeof(float);

	// conv5_5
	convolution_parallel << <numBlocks, threadInBlock, memsize_shared_memory >> > (dev_layer5_output_data2, image_size, image_size, num_filters5, dev_conv5_5_weights, kernel_size, 1, dev_layer5_output_data1);

	// Define CudaKernel settings
	threadInBlock.x = 16;
	threadInBlock.y = 16;
	threadInBlock.z = 1;
	numBlocks.x = (image_size + threadInBlock.x - 1) / threadInBlock.x;
	numBlocks.y = (image_size + threadInBlock.y - 1) / threadInBlock.y;
	numBlocks.z = num_filters5;

	// BatchNorm
	batch_normalization_parallel << <numBlocks, threadInBlock >> > (dev_layer5_output_data1, image_size, image_size, dev_conv5_batch5_beta, dev_conv5_batch5_gamma, dev_conv5_batch5_mean, dev_conv5_batch5_std, dev_layer5_output_data2);

	// ResidualConnection
	add_tensors_parallel << <numBlocks, threadInBlock >> > (dev_layer5_output_data2, dev_layer5_output_data3, image_size, image_size, dev_layer5_output_data1);

	// ReLU
	relu_parallel << <numBlocks, threadInBlock >> > (dev_layer5_output_data1, image_size, image_size, dev_layer5_output_data2);

	// Debug
	if (debug) {
		memsize_debug = image_size * image_size * num_filters5 * sizeof(float);
		debug_data = (float*)malloc(memsize_debug);
		cudaMemcpy(debug_data, dev_layer5_output_data2, memsize_debug, cudaMemcpyDeviceToHost);
		float max = FLT_MIN;
		float min = FLT_MAX;
		for (int i = 0; i < image_size * image_size * num_filters5; i++) {
			if (debug_data[i] > max) {
				max = debug_data[i];
			}
			if (debug_data[i] < min) {
				min = debug_data[i];
			}
		}
		printf("Output layer 5 (Identity block):\n");
		printf("Max value: %f\n", max);
		printf("Min value: %f\n", min);
	}

	// Final average pooling and fully connected layer
	// Device memory for linear_weights and linear_bias
	int memsize_fc_weights = num_filters5 * num_classes * sizeof(float);
	float* dev_fc_weights;
	cudaMalloc((void**)&dev_fc_weights, memsize_fc_weights);
	cudaMemcpy(dev_fc_weights, fc_weights, memsize_fc_weights, cudaMemcpyHostToDevice);

	int memsize_fc_bias = num_classes * sizeof(float);
	float* dev_fc_bias;
	cudaMalloc((void**)&dev_fc_bias, memsize_fc_bias);
	cudaMemcpy(dev_fc_bias, fc_bias, memsize_fc_bias, cudaMemcpyHostToDevice);

	// Device memory for intermediate tensors
	memsize_output_data = num_filters5 * sizeof(float);
	float* dev_flatten_data;
	cudaMalloc((void**)&dev_flatten_data, memsize_output_data);

	memsize_output_data = num_classes * sizeof(float);
	float* dev_logits_data;
	cudaMalloc((void**)&dev_logits_data, memsize_output_data);

	// Define CudaKernel settings
	threadInBlock.x = 16;
	threadInBlock.y = 16;
	threadInBlock.z = 1;
	numBlocks.x = (image_size + threadInBlock.x - 1) / threadInBlock.x;
	numBlocks.y = (image_size + threadInBlock.y - 1) / threadInBlock.y;
	numBlocks.z = num_filters5;
	memsize_shared_memory = threadInBlock.x * threadInBlock.y * threadInBlock.z * sizeof(float);

	// AveragePool
	average_pooling_parallel << <numBlocks, threadInBlock, memsize_shared_memory >> > (dev_layer5_output_data2, image_size, image_size, num_filters5, dev_flatten_data);

	// Debug
	if (debug) {
		memsize_debug = num_filters5 * sizeof(float);
		debug_data = (float*)malloc(memsize_debug);
		cudaMemcpy(debug_data, dev_flatten_data, memsize_debug, cudaMemcpyDeviceToHost);
		float max = FLT_MIN;
		float min = FLT_MAX;
		for (int i = 0; i < num_filters5; i++) {
			if (debug_data[i] > max) {
				max = debug_data[i];
			}
			if (debug_data[i] < min) {
				min = debug_data[i];
			}
		}
		printf("Average pooling layer:\n");
		printf("Max value: %f\n", max);
		printf("Min value: %f\n", min);
	}

	// Define CudaKernel settings
	int num_blocks = num_classes;
	int blockDim = 32;
	memsize_shared_memory = blockDim * sizeof(float);

	// Fully connected layer
	fully_connected_parallel << <num_blocks, blockDim, memsize_shared_memory >> > (dev_flatten_data, dev_fc_weights, dev_fc_bias, num_filters5, num_classes, dev_logits_data);

	// Debug
	if (debug) {
		memsize_debug = num_classes * sizeof(float);
		debug_data = (float*)malloc(memsize_debug);
		cudaMemcpy(debug_data, dev_logits_data, memsize_debug, cudaMemcpyDeviceToHost);
		float max = FLT_MIN;
		float min = FLT_MAX;
		for (int i = 0; i < num_filters5; i++) {
			if (i == 0) {
				printf("First value: % f\n", debug_data[i]);
			}
			if (debug_data[i] > max) {
				max = debug_data[i];
			}
			if (debug_data[i] < min) {
				min = debug_data[i];
			}
		}
		printf("Fully connected layer:\n");
		printf("Max value: %f\n", max);
		printf("Min value: %f\n", min);
	}

	// Softmax
	softmax_layer_parallel << <num_blocks, blockDim, memsize_shared_memory >> > (dev_logits_data, num_classes, dev_output_classes);

	// Move the output array from Device back to host.
	cudaMemcpy(output_classes, dev_output_classes, num_classes * sizeof(float), cudaMemcpyDeviceToHost);

	//Free allocated memory (device)
	cudaFree(dev_input_data);
	cudaFree(dev_output_classes);

	// conv1
	cudaFree(dev_conv1_weights);

	cudaFree(dev_conv1_batch1_beta);
	cudaFree(dev_conv1_batch1_gamma);
	cudaFree(dev_conv1_batch1_mean);
	cudaFree(dev_conv1_batch1_std);

	cudaFree(dev_layer1_output_data1);
	cudaFree(dev_layer1_output_data2);
	cudaFree(dev_layer1_output_data3);

	// conv2
	cudaFree(dev_conv2_1_weights);
	cudaFree(dev_conv2_2_weights);
	cudaFree(dev_conv2_3_weights);
	cudaFree(dev_conv2_4_weights);

	cudaFree(dev_conv2_batch1_beta);
	cudaFree(dev_conv2_batch1_gamma);
	cudaFree(dev_conv2_batch1_mean);
	cudaFree(dev_conv2_batch1_std);

	cudaFree(dev_conv2_batch2_beta);
	cudaFree(dev_conv2_batch2_gamma);
	cudaFree(dev_conv2_batch2_mean);
	cudaFree(dev_conv2_batch2_std);

	cudaFree(dev_conv2_batch3_beta);
	cudaFree(dev_conv2_batch3_gamma);
	cudaFree(dev_conv2_batch3_mean);
	cudaFree(dev_conv2_batch3_std);

	cudaFree(dev_conv2_batch4_beta);
	cudaFree(dev_conv2_batch4_gamma);
	cudaFree(dev_conv2_batch4_mean);
	cudaFree(dev_conv2_batch4_std);

	cudaFree(dev_layer2_output_data1);
	cudaFree(dev_layer2_output_data2);
	cudaFree(dev_layer2_output_data3);

	// conv3
	cudaFree(dev_conv3_1_weights);
	cudaFree(dev_conv3_2_weights);
	cudaFree(dev_conv3_3_weights);
	cudaFree(dev_conv3_4_weights);
	cudaFree(dev_conv3_5_weights);

	cudaFree(dev_conv3_batch1_beta);
	cudaFree(dev_conv3_batch1_gamma);
	cudaFree(dev_conv3_batch1_mean);
	cudaFree(dev_conv3_batch1_std);

	cudaFree(dev_conv3_batch2_beta);
	cudaFree(dev_conv3_batch2_gamma);
	cudaFree(dev_conv3_batch2_mean);
	cudaFree(dev_conv3_batch2_std);

	cudaFree(dev_conv3_batch3_beta);
	cudaFree(dev_conv3_batch3_gamma);
	cudaFree(dev_conv3_batch3_mean);
	cudaFree(dev_conv3_batch3_std);

	cudaFree(dev_conv3_batch4_beta);
	cudaFree(dev_conv3_batch4_gamma);
	cudaFree(dev_conv3_batch4_mean);
	cudaFree(dev_conv3_batch4_std);

	cudaFree(dev_conv3_batch5_beta);
	cudaFree(dev_conv3_batch5_gamma);
	cudaFree(dev_conv3_batch5_mean);
	cudaFree(dev_conv3_batch5_std);

	cudaFree(dev_layer3_output_data1);
	cudaFree(dev_layer3_output_data2);
	cudaFree(dev_layer3_output_data3);
	cudaFree(dev_layer3_output_data4);

	// conv4
	cudaFree(dev_conv4_1_weights);
	cudaFree(dev_conv4_2_weights);
	cudaFree(dev_conv4_3_weights);
	cudaFree(dev_conv4_4_weights);
	cudaFree(dev_conv4_5_weights);

	cudaFree(dev_conv4_batch1_beta);
	cudaFree(dev_conv4_batch1_gamma);
	cudaFree(dev_conv4_batch1_mean);
	cudaFree(dev_conv4_batch1_std);

	cudaFree(dev_conv4_batch2_beta);
	cudaFree(dev_conv4_batch2_gamma);
	cudaFree(dev_conv4_batch2_mean);
	cudaFree(dev_conv4_batch2_std);

	cudaFree(dev_conv4_batch3_beta);
	cudaFree(dev_conv4_batch3_gamma);
	cudaFree(dev_conv4_batch3_mean);
	cudaFree(dev_conv4_batch3_std);

	cudaFree(dev_conv4_batch4_beta);
	cudaFree(dev_conv4_batch4_gamma);
	cudaFree(dev_conv4_batch4_mean);
	cudaFree(dev_conv4_batch4_std);

	cudaFree(dev_conv4_batch5_beta);
	cudaFree(dev_conv4_batch5_gamma);
	cudaFree(dev_conv4_batch5_mean);
	cudaFree(dev_conv4_batch5_std);

	cudaFree(dev_layer4_output_data1);
	cudaFree(dev_layer4_output_data2);
	cudaFree(dev_layer4_output_data3);
	cudaFree(dev_layer4_output_data4);

	// conv5
	cudaFree(dev_conv5_1_weights);
	cudaFree(dev_conv5_2_weights);
	cudaFree(dev_conv5_3_weights);
	cudaFree(dev_conv5_4_weights);
	cudaFree(dev_conv5_5_weights);

	cudaFree(dev_conv5_batch1_beta);
	cudaFree(dev_conv5_batch1_gamma);
	cudaFree(dev_conv5_batch1_mean);
	cudaFree(dev_conv5_batch1_std);

	cudaFree(dev_conv5_batch2_beta);
	cudaFree(dev_conv5_batch2_gamma);
	cudaFree(dev_conv5_batch2_mean);
	cudaFree(dev_conv5_batch2_std);

	cudaFree(dev_conv5_batch3_beta);
	cudaFree(dev_conv5_batch3_gamma);
	cudaFree(dev_conv5_batch3_mean);
	cudaFree(dev_conv5_batch3_std);

	cudaFree(dev_conv5_batch4_beta);
	cudaFree(dev_conv5_batch4_gamma);
	cudaFree(dev_conv5_batch4_mean);
	cudaFree(dev_conv5_batch4_std);

	cudaFree(dev_conv5_batch5_beta);
	cudaFree(dev_conv5_batch5_gamma);
	cudaFree(dev_conv5_batch5_mean);
	cudaFree(dev_conv5_batch5_std);

	cudaFree(dev_layer5_output_data1);
	cudaFree(dev_layer5_output_data2);
	cudaFree(dev_layer5_output_data3);
	cudaFree(dev_layer5_output_data4);

	// Free allocated memory (host)
	// conv1
	for (int i = 0; i < num_filters1; i++) {
		free_tensor(&conv1_weights[i]);
	}

	free(conv1_batch1_beta);
	free(conv1_batch1_gamma);
	free(conv1_batch1_mean);
	free(conv1_batch1_std);

	// conv2_x
	for (int i = 0; i < num_filters2; i++) {
		free_tensor(&conv2_1_weights[i]);
		free_tensor(&conv2_2_weights[i]);
		free_tensor(&conv2_3_weights[i]);
		free_tensor(&conv2_4_weights[i]);
	}

	free(conv2_batch1_beta);
	free(conv2_batch1_gamma);
	free(conv2_batch1_mean);
	free(conv2_batch1_std);

	free(conv2_batch2_beta);
	free(conv2_batch2_gamma);
	free(conv2_batch2_mean);
	free(conv2_batch2_std);

	free(conv2_batch3_beta);
	free(conv2_batch3_gamma);
	free(conv2_batch3_mean);
	free(conv2_batch3_std);

	free(conv2_batch4_beta);
	free(conv2_batch4_gamma);
	free(conv2_batch4_mean);
	free(conv2_batch4_std);

	// conv3_x
	for (int i = 0; i < num_filters2; i++) {
		free_tensor(&conv3_1_weights[i]);
		free_tensor(&conv3_2_weights[i]);
		free_tensor(&conv3_3_weights[i]);
		free_tensor(&conv3_4_weights[i]);
		free_tensor(&conv3_5_weights[i]);
	}

	free(conv3_batch1_beta);
	free(conv3_batch1_gamma);
	free(conv3_batch1_mean);
	free(conv3_batch1_std);

	free(conv3_batch2_beta);
	free(conv3_batch2_gamma);
	free(conv3_batch2_mean);
	free(conv3_batch2_std);

	free(conv3_batch3_beta);
	free(conv3_batch3_gamma);
	free(conv3_batch3_mean);
	free(conv3_batch3_std);

	free(conv3_batch4_beta);
	free(conv3_batch4_gamma);
	free(conv3_batch4_mean);
	free(conv3_batch4_std);

	free(conv3_batch5_beta);
	free(conv3_batch5_gamma);
	free(conv3_batch5_mean);
	free(conv3_batch5_std);

	// conv4_x
	for (int i = 0; i < num_filters2; i++) {
		free_tensor(&conv4_1_weights[i]);
		free_tensor(&conv4_2_weights[i]);
		free_tensor(&conv4_3_weights[i]);
		free_tensor(&conv4_4_weights[i]);
		free_tensor(&conv4_5_weights[i]);
	}

	free(conv4_batch1_beta);
	free(conv4_batch1_gamma);
	free(conv4_batch1_mean);
	free(conv4_batch1_std);

	free(conv4_batch2_beta);
	free(conv4_batch2_gamma);
	free(conv4_batch2_mean);
	free(conv4_batch2_std);

	free(conv4_batch3_beta);
	free(conv4_batch3_gamma);
	free(conv4_batch3_mean);
	free(conv4_batch3_std);

	free(conv4_batch4_beta);
	free(conv4_batch4_gamma);
	free(conv4_batch4_mean);
	free(conv4_batch4_std);

	free(conv4_batch5_beta);
	free(conv4_batch5_gamma);
	free(conv4_batch5_mean);
	free(conv4_batch5_std);

	// conv5_x
	for (int i = 0; i < num_filters2; i++) {
		free_tensor(&conv5_1_weights[i]);
		free_tensor(&conv5_2_weights[i]);
		free_tensor(&conv5_3_weights[i]);
		free_tensor(&conv5_4_weights[i]);
		free_tensor(&conv5_5_weights[i]);
	}

	free(conv5_batch1_beta);
	free(conv5_batch1_gamma);
	free(conv5_batch1_mean);
	free(conv5_batch1_std);

	free(conv5_batch2_beta);
	free(conv5_batch2_gamma);
	free(conv5_batch2_mean);
	free(conv5_batch2_std);

	free(conv5_batch3_beta);
	free(conv5_batch3_gamma);
	free(conv5_batch3_mean);
	free(conv5_batch3_std);

	free(conv5_batch4_beta);
	free(conv5_batch4_gamma);
	free(conv5_batch4_mean);
	free(conv5_batch4_std);

	free(conv5_batch5_beta);
	free(conv5_batch5_gamma);
	free(conv5_batch5_mean);
	free(conv5_batch5_std);

	// Free fully connected layer weights and bias
	free(fc_weights);
	free(fc_bias);

	return cudaSuccess;
}