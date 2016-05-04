#include <chrono>
#include <climits>
#include <cuda_runtime.h>
#include <cuda_occupancy.h>
#include "ppm.hpp"
#include "add_weightedGPU.cuh"
#include "blurGPU.cuh"
#include "gauss_calculator.cuh"

//CPU Code
#include "unsharp_mask.cuh"

void unsharp_maskGPU(unsigned char *out, const unsigned char *in, const int blur_radius, const unsigned w, const unsigned h, const unsigned nchannels, bool gaussianblur) {

	float gridDimX = ceil(w / 32.0f); float gridDimY = ceil(h / 32.0f);
	dim3 blockSize(32, 32);
	dim3 gridSize(gridDimX, gridDimY);

	int imageSize = w * h * nchannels;

	unsigned char *d_out; cudaMalloc((void**)&d_out, imageSize);
	unsigned char *d_in1; cudaMalloc((void**)&d_in1, imageSize); cudaMemcpy(d_in1, in, imageSize, cudaMemcpyHostToDevice);

	if (gaussianblur){
		std::cout << "Started Unsharp Mask on GPU using Gauss Blur" << std::endl;
		float kernel_size = (blur_radius * 2) - 1;

		float *gaussKernel = calculate_gaussian(kernel_size, 10);

		unsigned char *d_blur; cudaMalloc((void**)&d_blur, imageSize);
		float *d_gauss; cudaMalloc((void**)&d_gauss, (kernel_size * kernel_size) * sizeof(float)); cudaMemcpy(d_gauss, gaussKernel, (kernel_size * kernel_size) * sizeof(float), cudaMemcpyHostToDevice);
		gaussBlurGPU<<<gridSize, blockSize>>>(d_blur, d_in1, w, h, nchannels, d_gauss, blur_radius);

		add_weightedGPU<<<gridSize, blockSize>>>(d_out, d_in1, 1.5f, d_blur, -0.5f, 0.0f, w, h, nchannels);
	}
	else{
		std::cout << "Started Unsharp Mask on GPU using Box Blur" << std::endl;
		unsigned char *d_blur1; cudaMalloc((void**)&d_blur1, imageSize);
		unsigned char *d_blur2; cudaMalloc((void**)&d_blur2, imageSize);
		unsigned char *d_blur3; cudaMalloc((void**)&d_blur3, imageSize);

		blurGPU<<<gridSize, blockSize>>>(d_blur1, d_in1, blur_radius, w, h, nchannels);
		blurGPU<<<gridSize, blockSize>>>(d_blur2, d_blur1, blur_radius, w, h, nchannels);
		blurGPU<<<gridSize, blockSize>>>(d_blur3, d_blur2, blur_radius, w, h, nchannels);

		add_weightedGPU<<<gridSize, blockSize>>>(d_out, d_in1, 1.5f, d_blur3, -0.5f, 0.0f, w, h, nchannels);
	}
	
	cudaMemcpy(out, d_out, imageSize, cudaMemcpyDeviceToHost);
	
}

// Apply an unsharp mask to the 24-bit PPM loaded from the file path of
// the first input argument; then write the sharpened output to the file path
// of the second argument. The third argument provides the blur radius.

int main(int argc, char *argv[])
{
	const char *ifilename = argc > 1 ? argv[1] : "../ghost-town-8k.ppm";
	const char *ofilename = argc > 2 ? argv[2] : "../out.ppm";
	const int blur_radius = argc > 3 ? std::atoi(argv[3]) : 5;
	const char *device = argc > 4 ? argv[4] : "gpu";
	const char *blur = argc > 5 ? argv[5] : "box";

	ppm img;
	std::vector<unsigned char> data_in, data_sharp;

	bool gauss;

	if (strcmp(blur, "box") == 0){
		gauss = false;
	}
	else if (strcmp(blur, "gauss") == 0){
		gauss = true;
	}

	std::cout << "Reading File..." << std::endl;

	img.read(ifilename, data_in);
	data_sharp.resize(img.w * img.h * img.nchannels);

	auto t1 = std::chrono::steady_clock::now();

	if (strcmp(device, "gpu") == 0){
		unsharp_maskGPU(data_sharp.data(), data_in.data(), blur_radius, img.w, img.h, img.nchannels, gauss);
	}
	else if (strcmp(device, "cpu") == 0){
		unsharp_mask(data_sharp.data(), data_in.data(), blur_radius, img.w, img.h, img.nchannels, gauss);
	}

	auto t2 = std::chrono::steady_clock::now();
	std::cout << std::chrono::duration<double>(t2 - t1).count() << " seconds.\n";

	if (blur_radius == 2 || blur_radius == 10){
		std::cout << "Writing File...." << std::endl;
		img.write(ofilename, data_sharp);
	}

	return 0;
}