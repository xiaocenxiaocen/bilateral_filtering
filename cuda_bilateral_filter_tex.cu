#include <cstdio>
#include <iostream>
#include <omp.h>
#include <time.h>
#include <cstdlib>
#include <cstring>
#include <assert.h>
#include <math.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include <pthread.h>
#include <complex>
//extern "C" {
//#include "lapacke.h"
//#include "lapacke_mangling.h"
//}
#include <mkl.h>

#include <mmintrin.h> 	// MMX
#include <xmmintrin.h>	// SSE
//#include <smmintrin.h>
//#include <nmmintrin.h>
#include <emmintrin.h>	// SSE2
#include <immintrin.h>	// AVX
//#include <intrin.h>

#include "cuda_common_api.h"
#include "cuda_common.cuh"

using namespace cv; // all the new API is put into "cv" namespace. Export its content
using namespace std;

typedef unsigned int uint;

const unsigned int Channel3 = 3;
const unsigned int BlockXcvtUchar3toRgba = 96;
const unsigned int BlockXcvtWrite = 32;
const unsigned int BlockYcvtUchar3toRgba = 2;

const unsigned int BlockXTransp = 16;
const unsigned int BlockYTransp = 16;

const float Epsilon = 1e-7;

__constant__ float gaussianKernelLookup[1024];
texture<uchar4, 2, cudaReadModeNormalizedFloat> rgbaTex;
cudaArray * d_array;

__device__ float EuclideanDistance(float4 lhs, float4 rhs, float tworangesqi)
{
	float sqdis = (lhs.x - rhs.x) * (lhs.x - rhs.x) +
		      (lhs.y - rhs.y) * (lhs.y - rhs.y) +
		      (lhs.z - rhs.z) * (lhs.z - rhs.z);
	return __expf(- sqdis * tworangesqi );
}

__device__ float4 cvt_uint_to_float4(uint c)
{
	float4 rgba;
	rgba.x = (c & 0xff) / 255.0f;
	rgba.y = ((c >> 8) & 0xff) / 255.0f;
	rgba.z = ((c >> 16) & 0xff) / 255.0f;
	rgba.w = ((c >> 24) & 0xff) / 255.0f;
	return rgba;	
}

__device__ uint cvt_float4_to_uint(float4 rgba)
{
	rgba.x = __saturatef(rgba.x);
	rgba.y = __saturatef(rgba.y);
	rgba.z = __saturatef(rgba.z);
	rgba.w = __saturatef(rgba.w);
	return (uint(rgba.w * 255) << 24) | (uint(rgba.z * 255) << 16) | (uint(rgba.y * 255) << 8) | (uint(rgba.x * 255));
}

inline int i_div_up(const int & divisor, const int & dividend)
{
	return ( divisor % dividend ) ? ( divisor / dividend + 1) : ( divisor / dividend );
}

__global__ void cvt_uchar3_to_rgba(const uchar * src, uint * dst, const int width32, const int height)
{
	
	unsigned int tx = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int ty = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int ltidx = threadIdx.x;
	unsigned int ltidy = threadIdx.y;

	__shared__ uchar smem[BlockYcvtUchar3toRgba][BlockXcvtUchar3toRgba];
	
	if(ty < height) {
		const uchar * src_ptr = src + ty * width32 * Channel3 + tx;
		smem[ltidy][ltidx] = *src_ptr;
	}
	__syncthreads();
	
	if(ty < height && ltidx < 32) {
		uint * dst_ptr = dst + ty * width32 + blockIdx.x * BlockXcvtWrite + ltidx;
		// bank conflict
		uchar x = smem[ltidy][Channel3 * ltidx];
		uchar y = smem[ltidy][Channel3 * ltidx + 1];
		uchar z = smem[ltidy][Channel3 * ltidx + 2];
		*dst_ptr = (uint(0x0) << 24) | (uint(z) << 16) | (uint(y) << 8) | uint(x);
	}
} 

__global__ void cvt_rgba_to_uchar3(const uint * src, uchar * dst, const int width32, const int height)
{
	unsigned int tx = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int ty = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int ltidx = threadIdx.x;
	unsigned int ltidy = threadIdx.y;
	__shared__ uchar smem[BlockYcvtUchar3toRgba][BlockXcvtUchar3toRgba];

	if(ty < height && ltidx < 32) {
		uint c = *(src + ty * width32 + blockIdx.x * BlockXcvtWrite + ltidx);
		// bank conflict
		smem[ltidy][Channel3 * ltidx    ] =  c & 0xff;
		smem[ltidy][Channel3 * ltidx + 1] = (c >> 8) & 0xff;
		smem[ltidy][Channel3 * ltidx + 2] = (c >> 16) & 0xff;
	}
	__syncthreads();
	
	if(ty < height && tx) {
		uchar * dst_ptr = dst + ty * width32 * Channel3 + tx;
		*dst_ptr = smem[ltidy][ltidx];
	}
	
}
__global__ void bilateral_filter_texture(uint * od, unsigned int width, unsigned int height, int kernel_radius, float rangeParams)
{
	unsigned int tx = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int ty = threadIdx.y + blockIdx.y * blockDim.y;
	
	if(tx >= width || ty >= height) return;

	float sum = 0.0f;
	float4 t = make_float4(0.f, 0.f, 0.f, 0.f);
	float4 center = tex2D(rgbaTex, tx, ty);

	for(int y = - kernel_radius; y <= kernel_radius; y++) {
		for(int x = - kernel_radius; x <= kernel_radius; x++) {
			float4 curPix = tex2D(rgbaTex, tx + x, ty + y);
			
			float factor = gaussianKernelLookup[y + kernel_radius] * gaussianKernelLookup[x + kernel_radius] * EuclideanDistance(curPix, center, rangeParams);
//			float factor = EuclideanDistance(curPix, center, rangeParams);
			t.x += factor * curPix.x;
			t.y += factor * curPix.y;
			t.z += factor * curPix.z;
			sum += factor;
		}
	}
	
	t.x /= sum; t.y /= sum; t.z /= sum;
	od[ty * width + tx] = cvt_float4_to_uint(t);	
}

extern "C"
void init_texture(int width, int height)
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
	cutilSafeCall(cudaMallocArray(&d_array, &channelDesc, width, height));	
}

extern "C"
void free_texture()
{
	cutilSafeCall(cudaFreeArray(d_array));
}

extern "C"
void get_gaussian_lookup(float spatialSigma, int radius)
{
	float gaussianLookup[2 * radius + 1];
	
	for(int i = 0; i < 2 * radius + 1; i++) {
		float x = i - radius;
		gaussianLookup[i] = expf(- x * x / (2.0 * spatialSigma * spatialSigma) );
	}	
	cutilSafeCall(cudaMemcpyToSymbol(gaussianKernelLookup, gaussianLookup, sizeof(float) * (2 * radius + 1)));
}

extern "C"
void bilateral_filter_gpu_texture(Mat& src, Mat& dst, float spatialSigma_x, float spatialSigma_y, float rangeParams)
{
	unsigned int width = src.cols;
	unsigned int height = src.rows;
	unsigned int width32 = (i_div_up(width, 32)) << 5;
	unsigned int height32 = (i_div_up(height, 32)) << 5;
	unsigned int cn = src.channels();	
	resize(src, src, Size(width32, height32));
	resize(dst, dst, Size(width32, height32));
	assert((src.cols & 0x1f) == 0 && cn == 3 && (src.rows & 0x1f) == 0);
	long int ibytes = cn * width32 * height32;
	float tworangesqi = 1.f / (2.f * rangeParams * rangeParams);
	assert(fabs(spatialSigma_x - spatialSigma_y) < Epsilon);
	int radius = static_cast<int>(3 * spatialSigma_x);

	float gputime = 0.0;

	cudaEvent_t start, end;
	create_event(&start);
	create_event(&end);

	record_event(start);

	get_gaussian_lookup(spatialSigma_x, radius);

	init_texture(width32, height32);
	
	long int totalBytes = 2 * ibytes;
	uchar * d_data;
	device_memory_allocate((void**)&d_data, totalBytes);
	uchar * d_src = d_data; uchar * d_dst = d_data + ibytes;

	long int totalMemVariabBytes = sizeof(uint) * width32 * height32;
	uint * d_rgba;
	device_memory_allocate((void**)&d_rgba, totalMemVariabBytes);

	uchar * h_src = src.ptr<uchar>(0);
	copy_to_device(d_src, h_src, ibytes);

	dim3 gridCvtUchar3toRgba(i_div_up(width32 * Channel3, BlockXcvtUchar3toRgba), i_div_up(height32, BlockYcvtUchar3toRgba), 1);
	dim3 blockCvtUchar3toRgba(BlockXcvtUchar3toRgba, BlockYcvtUchar3toRgba, 1);

	dim3 gridTransp_r2h(i_div_up(width32, BlockXTransp), i_div_up(height32, BlockYTransp), 1);
	dim3 blockTransp(BlockXTransp, BlockYTransp, 1);

	cvt_uchar3_to_rgba<<<gridCvtUchar3toRgba, blockCvtUchar3toRgba>>>(d_src, d_rgba, width32, height32);
	check_cuda_error("kernel cvt_uchar3_to_rgba");

	// from device global memory to texture memory
	cutilSafeCall(cudaMemcpyToArray(d_array, 0, 0, d_rgba, width32 * height32 * sizeof(uint), cudaMemcpyDeviceToDevice));  	
	cutilSafeCall(cudaBindTextureToArray(rgbaTex, d_array));

	// from texture memory to global memory
	bilateral_filter_texture<<<gridTransp_r2h, blockTransp>>>(d_rgba, width32, height32, radius, tworangesqi);

	cvt_rgba_to_uchar3<<<gridCvtUchar3toRgba, blockCvtUchar3toRgba>>>(d_rgba, d_dst, width32, height32);
	check_cuda_error("kernel cvt_rgba_to_uchar3");

	uchar * h_dst = dst.ptr<uchar>(0);
	copy_to_host(h_dst, d_dst, ibytes);
	resize(dst, dst, Size(width, height));

	record_event(end);
	sync_event(end);
	event_elapsed_time(&gputime, start, end);	
	destroy_event(start);
	destroy_event(end);
	fprintf(stdout, "INFO: elapsed time of cuda version bilateral filter with texture memory is %f\n", gputime * 0.001);	
	
	free_texture();
	free_device_memory(d_data);
	free_device_memory(d_rgba);
}

int main(int argc, char * argv[])
{
	if(argc < 5) {
		fprintf(stdout, "Usage: inputfile outputfile sigma range\n");
		return - 1;	
	}
	const char * inputimage = argv[1];
	const char * outputimage = argv[2];
	float sigma = atof(argv[3]);
	float range = atof(argv[4]);
	Mat src = imread(inputimage);
	if( !src.data) {
		fprintf(stderr, "ERROR: cannot open input image %s!\n", inputimage);
		exit(EXIT_FAILURE);
	}
	get_device_property(0);

	Mat dst(src.size(), src.type());
	bilateral_filter_gpu_texture(src, dst, sigma, sigma, range);
	imwrite(outputimage, dst);
	
	return 0;
}

