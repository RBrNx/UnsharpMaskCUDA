#include <cmath>

float gaussian(float x, float mu, float sigma){
	return exp(-(((x - mu) / (sigma)) * ((x - mu) / (sigma))) / 2.0);
}

float* calculate_gaussian(const int size, const float weight){
	float radius = ceil(size / 2);
	float sigma = radius / 2;
	float sum = 0;
	const double PI = 3.14159265359;
	float* kernel = new float[size * size];

	for (int y = 0; y < size; ++y){
		for (int x = 0; x < size; ++x){
			float a = gaussian(x, radius, sigma) * gaussian(y, radius, sigma);
			kernel[y * size + x] = a;
			sum += a;
		}
	}

	for (int x = 0; x < size; ++x){
		for (int y = 0; y < size; ++y){
			kernel[y * size + x] /= sum;
		}
	}

	return kernel;
}