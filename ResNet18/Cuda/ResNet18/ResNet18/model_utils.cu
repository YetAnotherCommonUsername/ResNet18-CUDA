#include "model_utils.cuh"

cudaError_t Conv2dWithCuda(struct tensor* input_tensor, struct tensor* kernels, int kernel_size, int stride, int num_channels, int num_filters, struct tensor* output_tensor) {
	// This function takes the struct input_image, the array kernel and perform the convolution using the GPU.
	// The resulting image will stored inside the struct 'output_image'.

	int memsize_input_tensor = input_tensor->col * input_tensor->row * input_tensor->depth * sizeof(float);
	int memsize_kernels = num_filters * kernel_size * kernel_size * num_channels * sizeof(float);
	int memsize_output_tensor = output_tensor->col * output_tensor->row * output_tensor->depth * sizeof(float);

	// Declaration of the input_array, kernel and output_array and move them to the GPU.
	float* dev_input_data;
	cudaMalloc((void**)&dev_input_data, memsize_input_tensor);
	cudaMemcpy(dev_input_data, input_tensor->data, memsize_input_tensor, cudaMemcpyHostToDevice);

	float* dev_kernels;
	cudaMalloc((void**)&dev_kernels, memsize_kernels);
	for (int i = 0; i < num_filters; i++) {
		cudaMemcpy(dev_kernels + i * kernel_size * kernel_size * num_channels, kernels[i].data, kernel_size * kernel_size * num_channels * sizeof(float), cudaMemcpyHostToDevice);
	}

	float* dev_output_data;
	cudaMalloc((void**)&dev_output_data, memsize_output_tensor);
	// No need to copy the output tensor to device before computation

	// Define CudaKernel settings.
	dim3 threadInBlock(8, 8, 16); // Adjust to suitable block size
	dim3 numBlocks;
	numBlocks.x = (input_tensor->col + threadInBlock.x - 1) / threadInBlock.x;
	numBlocks.y = (input_tensor->row + threadInBlock.y - 1) / threadInBlock.y;
	numBlocks.z = num_filters;
	int memsize_shared_memory = threadInBlock.x * threadInBlock.y * threadInBlock.z * sizeof(float);

	// Get the starting time.
	cudaError_t cudaStatus;

	// Launch the cuda kernel that performs the convolution.
	convolution_parallel << <numBlocks, threadInBlock, memsize_shared_memory >> > (dev_input_data, input_tensor->row, input_tensor->col, input_tensor->depth, dev_kernels, kernel_size, stride, dev_output_data);

	// Compute the elapsed time in ms.
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "convGPU launch failed: %s\n", cudaGetErrorString(cudaStatus));
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
			int output_image_idx = (kernel * (nrow / stride) * (ncol / stride)) + (blockIdx.y * blockDim.y + threadIdx.y) * (ncol / stride) + (blockIdx.x * blockDim.x + threadIdx.x);
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
