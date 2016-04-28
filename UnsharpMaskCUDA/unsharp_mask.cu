#include <chrono>
#include <climits>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_occupancy.h>
#include "ppm.hpp"

// Calculates the weighted sum of two arrays, in1 and in2 according
// to the formula: out(I) = saturate(in1(I)*alpha + in2(I)*beta + gamma)
template <typename T>
__global__ void add_weighted(unsigned char *out, unsigned char *in1, const T alpha, const unsigned char *in2, const T  beta, const T gamma, const unsigned w, const unsigned h, const unsigned nchannels) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;


	if (x < w && y < h){
		unsigned byte_offset = (y*w + x)*nchannels;

		T tmp = in1[byte_offset + 0] * alpha + in2[byte_offset + 0] * beta + gamma;
		out[byte_offset + 0] = tmp < 0 ? 0 : tmp > UCHAR_MAX ? UCHAR_MAX : tmp;

		tmp = in1[byte_offset + 1] * alpha + in2[byte_offset + 1] * beta + gamma;
		out[byte_offset + 1] = tmp < 0 ? 0 : tmp > UCHAR_MAX ? UCHAR_MAX : tmp;

		tmp = in1[byte_offset + 2] * alpha + in2[byte_offset + 2] * beta + gamma;
		out[byte_offset + 2] = tmp < 0 ? 0 : tmp > UCHAR_MAX ? UCHAR_MAX : tmp;
	}

}

// Averages the nsamples pixels within blur_radius of (x,y). Pixels which
// would be outside the image, replicate the value at the image border.
__device__ void pixel_average(unsigned char *out, const unsigned char *in, const int x, const int y, const int blur_radius, const unsigned w, const unsigned h, const unsigned nchannels) {
	float red_total = 0, green_total = 0, blue_total = 0;
	const unsigned nsamples = (blur_radius * 2 - 1) * (blur_radius * 2 - 1);

	for (int j = y - blur_radius + 1; j < y + blur_radius; ++j) {
		for (int i = x - blur_radius + 1; i < x + blur_radius; ++i) {
			const unsigned r_i = i < 0 ? 0 : i >= w ? w - 1 : i;
			const unsigned r_j = j < 0 ? 0 : j >= h ? h - 1 : j;
			unsigned byte_offset = (r_j*w + r_i)*nchannels;
			red_total += in[byte_offset + 0];
			green_total += in[byte_offset + 1];
			blue_total += in[byte_offset + 2];
		}
	}

	unsigned byte_offset = (y*w + x)*nchannels;
	out[byte_offset + 0] = red_total / nsamples;
	out[byte_offset + 1] = green_total / nsamples;
	out[byte_offset + 2] = blue_total / nsamples;
}

__global__ void blur(unsigned char *out, const unsigned char *in, const int blur_radius, const unsigned w, const unsigned h, const unsigned nchannels) {
	
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < w && y < h){
		pixel_average(out, in, x, y, blur_radius, w, h, nchannels);
	}
}

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
	
	//unsigned char *d_blur3; cudaMalloc((void**)&d_blur3, imageSize); cudaMemcpy(d_blur3, d_blur3, blur3.size(), cudaMemcpyHostToDevice);
	
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