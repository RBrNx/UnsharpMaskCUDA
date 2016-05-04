#include "blur.cuh"
#include "add_weighted.cuh"
#include <vector>

void unsharp_mask(unsigned char *out, const unsigned char *in,
                  const int blur_radius,
                  const unsigned w, const unsigned h, const unsigned nchannels, bool gaussianBlur)
{
	if (gaussianBlur){
		std::cout << "Started Unsharp Mask on CPU using Gauss Blur" << std::endl;
		std::vector<unsigned char> blur1;
		blur1.resize(w * h * nchannels);

		float kernel_size = (blur_radius * 2) - 1;

		float *gaussKernel = calculate_gaussian(kernel_size, 10);

		gaussBlur(blur1.data(), in, w, h, nchannels, gaussKernel, blur_radius);
		add_weighted(out, in, 1.5f, blur1.data(), -0.5f, 0.0f, w, h, nchannels);
	}
	else{
		std::cout << "Started Unsharp Mask on CPU using Box Blur" << std::endl;
		std::vector<unsigned char> blur1, blur2, blur3;

		blur1.resize(w * h * nchannels);
		blur2.resize(w * h * nchannels);
		blur3.resize(w * h * nchannels);

		blur(blur1.data(), in, blur_radius, w, h, nchannels);
		blur(blur2.data(), blur1.data(), blur_radius, w, h, nchannels);
		blur(blur3.data(), blur2.data(), blur_radius, w, h, nchannels);
		add_weighted(out, in, 1.5f, blur3.data(), -0.5f, 0.0f, w, h, nchannels);
	}
 
}
