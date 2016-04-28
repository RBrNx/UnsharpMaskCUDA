#include <chrono>
#include <climits>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_occupancy.h>
#include "ppm.hpp"
#include "add_weighted.cuh"
#include "blur.cuh"

void unsharp_mask(unsigned char *out, const unsigned char *in, const int blur_radius, const unsigned w, const unsigned h, const unsigned nchannels) {
	
	float gridDimX = ceil(w / 32.0f); float gridDimY = ceil(h / 32.0f);
	dim3 blockSize(32, 32);
	dim3 gridSize(gridDimX, gridDimY);

	std::vector<unsigned char> blur1, blur2, blur3;

	int imageSize = w * h * nchannels;

	blur1.resize(imageSize);
	blur2.resize(imageSize);
	blur3.resize(imageSize);

	unsigned char *d_blur1; cudaMalloc(&d_blur1, imageSize);
	unsigned char *d_blur2; cudaMalloc(&d_blur2, imageSize);
	unsigned char *d_blur3; cudaMalloc(&d_blur3, imageSize);
	unsigned char *d_out; cudaMalloc((void**)&d_out, imageSize);
	unsigned char *d_in1; cudaMalloc((void**)&d_in1, imageSize); cudaMemcpy(d_in1, in, imageSize, cudaMemcpyHostToDevice);

	blur<<<gridSize, blockSize>>>(d_blur1, d_in1, blur_radius, w, h, nchannels);
	blur<<<gridSize, blockSize>>>(d_blur2, d_blur1, blur_radius, w, h, nchannels);
	blur<<<gridSize, blockSize>>>(d_blur3, d_blur2, blur_radius, w, h, nchannels);
	
	add_weighted<<<gridSize, blockSize>>>(d_out, d_in1, 1.5f, d_blur3, -0.5f, 0.0f, w, h, nchannels);
	cudaMemcpy(out, d_out, imageSize, cudaMemcpyDeviceToHost);
}

// Apply an unsharp mask to the 24-bit PPM loaded from the file path of
// the first input argument; then write the sharpened output to the file path
// of the second argument. The third argument provides the blur radius.

int main(int argc, char *argv[])
{
	const char *ifilename = argc > 1 ? argv[1] : "../lena.ppm";
	const char *ofilename = argc > 2 ? argv[2] : "../out.ppm";
	const int blur_radius = argc > 3 ? std::atoi(argv[3]) : 5;

	ppm img;
	std::vector<unsigned char> data_in, data_sharp;

	img.read(ifilename, data_in);
	data_sharp.resize(img.w * img.h * img.nchannels);

	auto t1 = std::chrono::steady_clock::now();

	unsharp_mask(data_sharp.data(), data_in.data(), blur_radius, img.w, img.h, img.nchannels);

	auto t2 = std::chrono::steady_clock::now();
	std::cout << std::chrono::duration<double>(t2 - t1).count() << " seconds.\n";

	img.write(ofilename, data_sharp);

	return 0;
}