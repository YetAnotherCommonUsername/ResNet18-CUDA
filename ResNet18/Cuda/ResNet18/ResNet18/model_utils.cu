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

	// Define CudaKernel settings.
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

/*
// Riparametrizzazione convoluzione
if ((row >= pad && row < (nrow - pad)) && (col >= pad && col < (ncol - pad))) {
	int start_idx = -kernel_size / 2;
	int result = 0;
	int input_image_idx = (nrow * ncol) * channel;
	int kernel_idx = (kernel_size * kernel_size) * channel;

	for (int i = start_idx; i <= kernel_size / 2; i++) {
		for (int j = start_idx; j <= kernel_size / 2; j++) {
			result += (kernel[kernel_idx] * input_tensor[input_image_idx + ncol * (row + i) + (col + j)]);
			kernel_idx++;
		}
	}
	tmp_conv_channels[channel] = result;
}
*/

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

	