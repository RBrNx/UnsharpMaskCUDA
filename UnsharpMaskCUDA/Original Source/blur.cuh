// Averages the nsamples pixels within blur_radius of (x,y). Pixels which
// would be outside the image, replicate the value at the image border.
void pixel_average(      unsigned char *out,
                   const unsigned char *in,
                   const int x, const int y, const int blur_radius,
                   const unsigned w, const unsigned h, const unsigned nchannels)
{
  float red_total = 0, green_total = 0, blue_total = 0;
  const unsigned nsamples = (blur_radius*2-1) * (blur_radius*2-1);
  for (int j = y-blur_radius+1; j < y+blur_radius; ++j) {
    for (int i = x-blur_radius+1; i < x+blur_radius; ++i) {
      const unsigned r_i = i < 0 ? 0 : i >= w ? w-1 : i;
      const unsigned r_j = j < 0 ? 0 : j >= h ? h-1 : j;
      unsigned byte_offset = (r_j*w+r_i)*nchannels;
      red_total   += in[byte_offset+0];
      green_total += in[byte_offset+1];
      blue_total  += in[byte_offset+2];
    }
  }

  unsigned byte_offset = (y*w+x)*nchannels;
  out[byte_offset+0] =   red_total/nsamples;
  out[byte_offset+1] = green_total/nsamples;
  out[byte_offset+2] =  blue_total/nsamples;
}

void blur(unsigned char *out, const unsigned char *in,
          const int blur_radius,
          const unsigned w, const unsigned h, const unsigned nchannels)
{
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      pixel_average(out,in,x,y,blur_radius,w,h,nchannels);
    }
  }
}


void gaussPixel_average(unsigned char *out, const unsigned char *in, const int x, const int y, const unsigned w, const unsigned h, const unsigned nchannels, const float *gausskernel, const int blur_radius){
	float red_total = 0, green_total = 0, blue_total = 0;
	const unsigned nsamples = (blur_radius * 2 - 1) * (blur_radius * 2 - 1);

	for (int j = y - blur_radius + 1, ky = 0; j < y + blur_radius; ++j, ++ky) {
		for (int i = x - blur_radius + 1, kx = 0; i < x + blur_radius; ++i, ++kx) {
			const unsigned r_i = i < 0 ? 0 : i >= w ? w - 1 : i;
			const unsigned r_j = j < 0 ? 0 : j >= h ? h - 1 : j;
			unsigned byte_offset = (r_j*w + r_i)*nchannels;
			red_total += (in[byte_offset + 0] * gausskernel[ky * ((blur_radius * 2) - 1) + kx]);
			green_total += in[byte_offset + 1] * gausskernel[ky * ((blur_radius * 2) - 1) + kx];
			blue_total += in[byte_offset + 2] * gausskernel[ky * ((blur_radius * 2) - 1) + kx];
		}
	}

	unsigned byte_offset = (y*w + x)*nchannels;
	out[byte_offset + 0] = red_total;// / nsamples;
	out[byte_offset + 1] = green_total;// / nsamples;
	out[byte_offset + 2] = blue_total;// / nsamples;
}

void gaussBlur(unsigned char *out, const unsigned char *in, const unsigned w, const unsigned h, const unsigned nchannels, const float *gauss, const int blur_radius){
	for (int y = 0; y < h; ++y) {
		for (int x = 0; x < w; ++x) {
			gaussPixel_average(out, in, x, y,w, h, nchannels, gauss, blur_radius);
		}
	}
}

