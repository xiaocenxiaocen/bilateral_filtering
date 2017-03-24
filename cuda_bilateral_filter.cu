/* @file: bilateral_filter.cu
 * @brief: reference, Yang Q., Tan K., Ahuja N., Real-Time O(1) Bilateral Filtering, CVPR, 2009
 * @author: xiaocen
 * @date: 2017.03.15
 */
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

const unsigned int BlockXTransp = 32;
const unsigned int BlockYTransp = 32;

__global__ void transpose(uint * id, uint * od, const int width, const int height)
{
	unsigned int tx = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int ty = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int ltidx = threadIdx.x;
	unsigned int ltidy = threadIdx.y;
	unsigned int inputIdx = ty * width + tx;

	__shared__ uint smem[BlockYTransp][BlockXTransp];
	if(tx < width && ty < height) {
		smem[ltidy][ltidx] = id[inputIdx];
	}
	__syncthreads();

	tx = threadIdx.y + blockIdx.x * blockDim.x;
	ty = threadIdx.x + blockIdx.y * blockDim.y;
	unsigned int outputIdx = tx * height + ty;
	if(tx < width && ty < height) {
		od[outputIdx] = smem[ltidx][ltidy];	
	}
}

__device__ __constant__ float d_bx[4];
__device__ __constant__ float d_bx_[4];
__device__ __constant__ float d_ax[4];

__device__ __constant__ float d_by[4];
__device__ __constant__ float d_by_[4];
__device__ __constant__ float d_ay[4];

static const double deriche4[] = {1.64847058, -0.64905727, 3.59281342, -0.23814081, 0.63750105, 2.01124176, 1.760522, 1.70738557};
const int Radius = 4;

inline void prepare_coeffs_recursive_gaussian_x(double sigma)
{
	float _b[4];
	float _a[4];
	float _b_[4];
	complex<double> poles[4];
	complex<double> ai[4];
	const double * c = deriche4;
	const double * s = deriche4 + 2;
	const double * w = deriche4 + 4;
	const double * b = deriche4 + 6;
	for(int i = 0; i < 2; i++) {
		poles[2 * i    ] = exp(complex<double>(- b[i] / sigma,  w[i] / sigma));
		poles[2 * i + 1] = exp(complex<double>(- b[i] / sigma, -w[i] / sigma));
		ai[2 * i    ] = complex<double>(0.5 * c[i], - 0.5 * s[i]);
		ai[2 * i + 1] = complex<double>(0.5 * c[i],   0.5 * s[i]);
	}
	complex<double> A[Radius][Radius];
	complex<double> rhs[Radius];
	int ipiv[Radius];
	for(int i = 0; i < Radius; i++) {
		A[i][0] = 1.0 / poles[i];
		rhs[i] = -1.0;
		if(i != 0) ai[i] *= 1.0 - poles[0] * A[i][0];
		for(int j = 1; j < Radius; j++) {
			if((j + 1) % 2 == 0) {
				A[i][j] = A[i][(j - 1) / 2] * A[i][(j - 1) / 2];
			} else {
				A[i][j] = A[i][(j - 1) / 2] * A[i][j / 2];
			}
			if(i != j) ai[i] *= 1.0 - poles[j] * A[i][0];
		}
	}
	int ret;
	if((ret = LAPACKE_zgetrf(LAPACK_ROW_MAJOR, Radius, Radius, reinterpret_cast<lapack_complex_double*>(A[0]), Radius, ipiv)) != 0) {
		fprintf(stderr, "ERROR: lapack routine LAPACKE_zgetrf() for solving a of simga = %f failed, ret = %d!\n", sigma, ret);
		exit(-1);
	}
	if((ret = LAPACKE_zgetrs(LAPACK_ROW_MAJOR, 'N', Radius, 1, reinterpret_cast<lapack_complex_double*>(A[0]), Radius, ipiv, reinterpret_cast<lapack_complex_double*>(rhs), 1)) != 0) {
		fprintf(stderr, "ERROR: lapack routine LAPACKE_zgetrs() for solving a of sigma = %f failed, ret = %d!\n", sigma, ret);
		exit(-1);
	}
	for(int i = 0; i < Radius; i++) {
		A[i][0] = 1.0;
		A[i][1] = 1.0 / poles[i];
		A[i][2] = A[i][1] * A[i][1];
		A[i][3] = A[i][2] * A[i][1];
	}
	if((ret = LAPACKE_zgetrf(LAPACK_ROW_MAJOR, Radius, Radius, reinterpret_cast<lapack_complex_double*>(A[0]), Radius, ipiv)) != 0) {
		fprintf(stderr, "ERROR: lapack routine LAPACKE_zgetrf() for solving b of simga = %f failed, ret = %d!\n", sigma, ret);
		exit(-1);
	}
	if((ret = LAPACKE_zgetrs(LAPACK_ROW_MAJOR, 'N', Radius, 1, reinterpret_cast<lapack_complex_double*>(A[0]), Radius, ipiv, reinterpret_cast<lapack_complex_double*>(ai), 1)) != 0) {
		fprintf(stderr, "ERROR: lapack routine LAPACKE_zgetrs() for solving b of sigma = %f failed, ret = %d!\n", sigma, ret);
		exit(-1);
	}
	for(int i = 0; i < Radius; i++) {
		_a[i] = real(rhs[i]);
		_b[i] = real(ai[i]) / (sqrt(2.0 * M_PI ) * sigma);
	}
	_b_[0] = _b[1] - _b[0] * _a[0];
	_b_[1] = _b[2] - _b[0] * _a[1];
	_b_[2] = _b[3] - _b[0] * _a[2];
	_b_[3] = - _b[0] * _a[3];

	cutilSafeCall( cudaMemcpyToSymbol(d_bx, (void*)_b, sizeof(float) * 4) );
	cutilSafeCall( cudaMemcpyToSymbol(d_bx_, (void*)_b_, sizeof(float) * 4) );
	cutilSafeCall( cudaMemcpyToSymbol(d_ax, (void*)_a, sizeof(float) * 4) );
}

inline void prepare_coeffs_recursive_gaussian_y(double sigma)
{
	float _b[4];
	float _a[4];
	float _b_[4];
	complex<double> poles[4];
	complex<double> ai[4];
	const double * c = deriche4;
	const double * s = deriche4 + 2;
	const double * w = deriche4 + 4;
	const double * b = deriche4 + 6;
	for(int i = 0; i < 2; i++) {
		poles[2 * i    ] = exp(complex<double>(- b[i] / sigma,  w[i] / sigma));
		poles[2 * i + 1] = exp(complex<double>(- b[i] / sigma, -w[i] / sigma));
		ai[2 * i    ] = complex<double>(0.5 * c[i], - 0.5 * s[i]);
		ai[2 * i + 1] = complex<double>(0.5 * c[i],   0.5 * s[i]);
	}
	complex<double> A[Radius][Radius];
	complex<double> rhs[Radius];
	int ipiv[Radius];
	for(int i = 0; i < Radius; i++) {
		A[i][0] = 1.0 / poles[i];
		rhs[i] = -1.0;
		if(i != 0) ai[i] *= 1.0 - poles[0] * A[i][0];
		for(int j = 1; j < Radius; j++) {
			if((j + 1) % 2 == 0) {
				A[i][j] = A[i][(j - 1) / 2] * A[i][(j - 1) / 2];
			} else {
				A[i][j] = A[i][(j - 1) / 2] * A[i][j / 2];
			}
			if(i != j) ai[i] *= 1.0 - poles[j] * A[i][0];
		}
	}
	int ret;
	if((ret = LAPACKE_zgetrf(LAPACK_ROW_MAJOR, Radius, Radius, reinterpret_cast<lapack_complex_double*>(A[0]), Radius, ipiv)) != 0) {
		fprintf(stderr, "ERROR: lapack routine LAPACKE_zgetrf() for solving a of simga = %f failed, ret = %d!\n", sigma, ret);
		exit(-1);
	}
	if((ret = LAPACKE_zgetrs(LAPACK_ROW_MAJOR, 'N', Radius, 1, reinterpret_cast<lapack_complex_double*>(A[0]), Radius, ipiv, reinterpret_cast<lapack_complex_double*>(rhs), 1)) != 0) {
		fprintf(stderr, "ERROR: lapack routine LAPACKE_zgetrs() for solving a of sigma = %f failed, ret = %d!\n", sigma, ret);
		exit(-1);
	}
	for(int i = 0; i < Radius; i++) {
		A[i][0] = 1.0;
		A[i][1] = 1.0 / poles[i];
		A[i][2] = A[i][1] * A[i][1];
		A[i][3] = A[i][2] * A[i][1];
	}
	if((ret = LAPACKE_zgetrf(LAPACK_ROW_MAJOR, Radius, Radius, reinterpret_cast<lapack_complex_double*>(A[0]), Radius, ipiv)) != 0) {
		fprintf(stderr, "ERROR: lapack routine LAPACKE_zgetrf() for solving b of simga = %f failed, ret = %d!\n", sigma);
		exit(-1);
	}
	if((ret = LAPACKE_zgetrs(LAPACK_ROW_MAJOR, 'N', Radius, 1, reinterpret_cast<lapack_complex_double*>(A[0]), Radius, ipiv, reinterpret_cast<lapack_complex_double*>(ai), 1)) != 0) {
		fprintf(stderr, "ERROR: lapack routine LAPACKE_zgetrs() for solving b of sigma = %f failed, ret = %d!\n", sigma, ret);
		exit(-1);
	}
	for(int i = 0; i < Radius; i++) {
		_a[i] = real(rhs[i]);
		_b[i] = real(ai[i]) / (sqrt(2.0 * M_PI ) * sigma);
	}
	_b_[0] = _b[1] - _b[0] * _a[0];
	_b_[1] = _b[2] - _b[0] * _a[1];
	_b_[2] = _b[3] - _b[0] * _a[2];
	_b_[3] = - _b[0] * _a[3];

	cutilSafeCall( cudaMemcpyToSymbol(d_by, (void*)_b, sizeof(float) * 4) );
	cutilSafeCall( cudaMemcpyToSymbol(d_by_, (void*)_b_, sizeof(float) * 4) );
	cutilSafeCall( cudaMemcpyToSymbol(d_ay, (void*)_a, sizeof(float) * 4) );
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

inline __device__ void advance( float4 * queue, const int nums )
{
	for(int i = 0; i < nums; i++) {
		queue[i] = queue[i + 1];
	}
}

inline __device__ void advance_shared( float4 * queue, const int nums )
{
	for(int i = 0; i < nums; i++) {
		queue[i * blockDim.x] = queue[(i + 1) * blockDim.x];
	}
}

const int BlockXRecurGaussianRgba = 32;

__global__ void recursive_gaussian_rgba_col(uint * id, uint * od, uint * temp, const int width, const int height)
{
	unsigned int tx = threadIdx.x + blockIdx.x * blockDim.x;

	if(tx < width) {
	od += tx;
	id += tx;
	temp += tx;

	float4 queue[4];
	float4 queue_h[4];
	
	// forward pass
	int processIdx = 0;
	for(int row = 0; row < height; row++, processIdx += width) {
		float4 x_out = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		if(row < 4) {
			queue[row] = cvt_uint_to_float4(id[processIdx]);
		} else {
			advance( queue, 3 );
			queue[3] = cvt_uint_to_float4(id[processIdx]);
		}
		#pragma unroll 4
		for(int k = 0; k < 4; k++) {
			int index_x = row < 4 ? row - k : 3 - k;
			if(row >= k) {
				x_out.x += queue[index_x].x * d_by[k];
				x_out.y += queue[index_x].y * d_by[k];
				x_out.z += queue[index_x].z * d_by[k];
			} else {
				x_out.x += queue[0].x * d_by[k];
				x_out.y += queue[0].y * d_by[k];		
				x_out.z += queue[0].z * d_by[k];
			}
		//	x_out.x += row >= k ? queue[index_x].x * d_by[k] : queue[0].x * d_by[k];
		//	x_out.y += row >= k ? queue[index_x].y * d_by[k] : queue[0].y * d_by[k];
		//	x_out.z += row >= k ? queue[index_x].z * d_by[k] : queue[0].z * d_by[k];

			int index_y = row <= 4 ? row - k - 1 : 3 - k;
			if(row >= k + 1) {
				x_out.x -= queue_h[index_y].x * d_ay[k];
				x_out.y -= queue_h[index_y].y * d_ay[k];
				x_out.z -= queue_h[index_y].z * d_ay[k];
			} else {
				x_out.x -= queue[0].x * d_ay[k];
				x_out.y -= queue[0].y * d_ay[k];	
				x_out.z -= queue[0].z * d_ay[k];	
			}

		//	x_out.x -= row >= k + 1 ? queue_h[index_y].x * d_ay[k] : queue[0].x * d_ay[k];	
		//	x_out.y -= row >= k + 1 ? queue_h[index_y].y * d_ay[k] : queue[0].y * d_ay[k];
		//	x_out.z -= row >= k + 1 ? queue_h[index_y].z * d_ay[k] : queue[0].z * d_ay[k];
		}
		if(row < 4) {
			queue_h[row] = x_out;
		} else {
			advance( queue_h, 3 );
			queue_h[3] = x_out;
		}
		od[processIdx] = cvt_float4_to_uint(x_out);
	}	

	processIdx -= width;
	// backward pass
	for(int row = height - 1; row >= 0; row--, processIdx -= width) { 
		float4 x_out = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		if(row == height - 1) queue[0] = cvt_uint_to_float4(id[processIdx]);
		#pragma unroll 4
		for(int k = 0; k < 4; k++) {
			int index = row >= height - 4 ? height - row - k - 2 : 3 - k;
			if(row < height - k - 1) {
				x_out.x += queue[index].x * d_by_[k] - queue_h[index].x * d_ay[k];
				x_out.y += queue[index].y * d_by_[k] - queue_h[index].y * d_ay[k];
				x_out.z += queue[index].z * d_by_[k] - queue_h[index].z * d_ay[k];
			} else {
				x_out.x += queue[0].x * (d_by_[k] - d_ay[k]);
				x_out.y += queue[0].y * (d_by_[k] - d_ay[k]);
				x_out.z += queue[0].z * (d_by_[k] - d_ay[k]);
			}
		
	//		x_out.x += row < height - k - 1 ? queue[index].x * d_by_[k] : queue[0].x * d_by_[k];
	//		x_out.y += row < height - k - 1 ? queue[index].y * d_by_[k] : queue[0].y * d_by_[k];
	//		x_out.z += row < height - k - 1 ? queue[index].z * d_by_[k] : queue[0].z * d_by_[k];
	//		
	//		x_out.x -= row < height - k - 1 ? queue_h[index].x * d_ay[k] : queue[0].x * d_ay[k];
	//		x_out.y -= row < height - k - 1 ? queue_h[index].y * d_ay[k] : queue[0].y * d_ay[k];
	//		x_out.z -= row < height - k - 1 ? queue_h[index].z * d_ay[k] : queue[0].z * d_ay[k];
		}
		if(row >= height - 4) {
			queue_h[height - 1- row] = x_out;
			if(row < height - 1) {
				queue[height - 1 - row] = cvt_uint_to_float4(id[processIdx]);
			}	
		} else {
			advance( queue_h, 3 );
			queue_h[3] = x_out;

			advance( queue, 3 );
			queue[3] = cvt_uint_to_float4(id[processIdx]);
		}
		temp[processIdx] = cvt_float4_to_uint(x_out);
		float4 x_fwd = cvt_uint_to_float4(od[processIdx]);
		x_out.x += x_fwd.x; x_out.y += x_fwd.y; x_out.z += x_fwd.z;	
		od[processIdx] = cvt_float4_to_uint(x_out);
	}
	}
}

__global__ void recursive_gaussian_rgba_col_fwdpass(uint * id, uint * od, const int width, const int height)
{
	unsigned int tx = threadIdx.x + blockIdx.x * blockDim.x;
	__shared__ float4 queue_smem[4 * BlockXRecurGaussianRgba];
	__shared__ float4 queue_h_smem[4 * BlockXRecurGaussianRgba];

	if(tx < width) {
	od += tx;
	id += tx;

	float4 * queue = queue_smem + threadIdx.x;
	float4 * queue_h = queue_h_smem + threadIdx.x;
//	float4 queue[4];
//	float4 queue_h[4];
	
	// forward pass
	int processIdx = 0;
	for(int row = 0; row < height; row++, processIdx += width) {
		float4 x_out = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		if(row < 4) {
			queue[row * blockDim.x] = cvt_uint_to_float4(id[processIdx]);
		//	queue[row] = cvt_uint_to_float4(id[processIdx]);
		} else {
		//	advance( queue, 3 );
			advance_shared( queue, 3 );
			queue[3 * blockDim.x] = cvt_uint_to_float4(id[processIdx]);
		//	queue[3] = cvt_uint_to_float4(id[processIdx]);
		}
		#pragma unroll 4
		for(int k = 0; k < 4; k++) {
			int index_x = row < 4 ? row - k : 3 - k;
			if(row >= k) {
				x_out.x += queue[index_x * blockDim.x].x * d_by[k];
				x_out.y += queue[index_x * blockDim.x].y * d_by[k];
				x_out.z += queue[index_x * blockDim.x].z * d_by[k];
		//		x_out.x += queue[index_x].x * d_by[k];
		//		x_out.y += queue[index_x].y * d_by[k];
		//		x_out.z += queue[index_x].z * d_by[k];
			} else {
				x_out.x += queue[0].x * d_by[k];
				x_out.y += queue[0].y * d_by[k];		
				x_out.z += queue[0].z * d_by[k];
			}
		//	x_out.x += row >= k ? queue[index_x * blockDim.x].x * d_by[k] : queue[0].x * d_by[k];
		//	x_out.y += row >= k ? queue[index_x * blockDim.x].y * d_by[k] : queue[0].y * d_by[k];
		//	x_out.z += row >= k ? queue[index_x * blockDim.x].z * d_by[k] : queue[0].z * d_by[k];

			int index_y = row <= 4 ? row - k - 1 : 3 - k;
			if(row >= k + 1) {
				x_out.x -= queue_h[index_y * blockDim.x].x * d_ay[k];
				x_out.y -= queue_h[index_y * blockDim.x].y * d_ay[k];
				x_out.z -= queue_h[index_y * blockDim.x].z * d_ay[k];
			//	x_out.x -= queue_h[index_y].x * d_ay[k];
			//      x_out.y -= queue_h[index_y].y * d_ay[k];
			//      x_out.z -= queue_h[index_y].z * d_ay[k];
			} else {
				x_out.x -= queue[0].x * d_ay[k];
				x_out.y -= queue[0].y * d_ay[k];	
				x_out.z -= queue[0].z * d_ay[k];	
			}

		//	x_out.x -= row >= k + 1 ? queue_h[index_y].x * d_ay[k] : queue[0].x * d_ay[k];	
		//	x_out.y -= row >= k + 1 ? queue_h[index_y].y * d_ay[k] : queue[0].y * d_ay[k];
		//	x_out.z -= row >= k + 1 ? queue_h[index_y].z * d_ay[k] : queue[0].z * d_ay[k];
		}
		if(row < 4) {
			queue_h[row * blockDim.x] = x_out;
	//		queue_h[row] = x_out;
		} else {
//			advance( queue_h, 3 );
			advance_shared( queue_h, 3 );
			queue_h[3 * blockDim.x] = x_out;
//			queue_h[3] = x_out;
		}
		od[processIdx] = cvt_float4_to_uint(x_out);
	}
	}
}

__global__ void recursive_gaussian_rgba_col_revpass(uint * id, uint * od, const int width, const int height)
{
	unsigned int tx = threadIdx.x + blockIdx.x * blockDim.x;
//	__shared__ float4 queue_smem[4 * BlockXRecurGaussianRgba];
//	__shared__ float4 queue_h_smem[4 * BlockXRecurGaussianRgba];


	if(tx < width) {
	od += tx;
	id += tx;

//	float4 * queue = queue_smem + threadIdx.x;
//	float4 * queue_h = queue_h_smem + threadIdx.x;
	float4 queue[4];
	float4 queue_h[4];

	int processIdx = (height - 1) * width;
	// backward pass
	for(int row = height - 1; row >= 0; row--, processIdx -= width) { 
		float4 x_out = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		if(row == height - 1) queue[0] = cvt_uint_to_float4(id[processIdx]);
		#pragma unroll 4
		for(int k = 0; k < 4; k++) {
			int index = row >= height - 4 ? height - row - k - 2 : 3 - k;
			if(row < height - k - 1) {
				x_out.x += queue[index].x * d_by_[k] - queue_h[index].x * d_ay[k];
				x_out.y += queue[index].y * d_by_[k] - queue_h[index].y * d_ay[k];
				x_out.z += queue[index].z * d_by_[k] - queue_h[index].z * d_ay[k];
			//	x_out.x += queue[index * blockDim.x].x * d_by_[k] - queue_h[index * blockDim.x].x * d_ay[k];
			//	x_out.y += queue[index * blockDim.x].y * d_by_[k] - queue_h[index * blockDim.x].y * d_ay[k];
			//	x_out.z += queue[index * blockDim.x].z * d_by_[k] - queue_h[index * blockDim.x].z * d_ay[k];
			} else {
				x_out.x += queue[0].x * (d_by_[k] - d_ay[k]);
				x_out.y += queue[0].y * (d_by_[k] - d_ay[k]);
				x_out.z += queue[0].z * (d_by_[k] - d_ay[k]);
			}
		
	//		x_out.x += row < height - k - 1 ? queue[index].x * d_by_[k] : queue[0].x * d_by_[k];
	//		x_out.y += row < height - k - 1 ? queue[index].y * d_by_[k] : queue[0].y * d_by_[k];
	//		x_out.z += row < height - k - 1 ? queue[index].z * d_by_[k] : queue[0].z * d_by_[k];
	//		
	//		x_out.x -= row < height - k - 1 ? queue_h[index].x * d_ay[k] : queue[0].x * d_ay[k];
	//		x_out.y -= row < height - k - 1 ? queue_h[index].y * d_ay[k] : queue[0].y * d_ay[k];
	//		x_out.z -= row < height - k - 1 ? queue_h[index].z * d_ay[k] : queue[0].z * d_ay[k];
		}
		if(row >= height - 4) {
			queue_h[height - 1- row] = x_out;
	//		queue_h[(height - row) * blockDim.x - blockDim.x] = x_out;
			if(row < height - 1) {
				queue[height - 1 - row] = cvt_uint_to_float4(id[processIdx]);
	//			queue[(height - row) * blockDim.x - blockDim.x] = x_out;
			}	
		} else {
			advance( queue_h, 3 );
	//		advance_shared( queue_h, 3 );
			queue_h[3] = x_out;
	//		queue_h[3 * blockDim.x] = x_out;

			advance( queue, 3 );
	//		advance_shared( queue, 3 );
			queue[3] = cvt_uint_to_float4(id[processIdx]);
	//		queue[3 * blockDim.x] = cvt_uint_to_float4(id[processIdx]);
		}
		od[processIdx] = cvt_float4_to_uint(x_out);
	}
	}
}

__global__ void recursive_gaussian_rgba_sum(uint * id, uint * od, const int width, const int height)
{
	unsigned int tx = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int ty = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int index = ty * width + tx;

	if(tx < width && ty < height) {
		float4 float_in = cvt_uint_to_float4(id[index]);
		float4 float_out = cvt_uint_to_float4(od[index]);
		float_out.x += float_in.x; float_out.y += float_in.y; float_out.z += float_in.z;
		od[index] = cvt_float4_to_uint(float_out);
	}
}

__global__ void recursive_gaussian_rgba_row(uint * id, uint * od, uint * temp, const int width, const int height) 
{
	unsigned int tx = threadIdx.x + blockIdx.x * blockDim.x;

	if(tx < width) {
	od += tx;
	id += tx;
	temp += tx;

	float4 queue[4];
	float4 queue_h[4];
	
	// forward pass
	int processIdx = 0;
	for(int row = 0; row < height; row++, processIdx += width) {
		float4 x_out = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		if(row < 4) {
			queue[row] = cvt_uint_to_float4(id[processIdx]);
		} else {
			advance( queue, 3 );
			queue[3] = cvt_uint_to_float4(id[processIdx]);
		}
		#pragma unroll 4
		for(int k = 0; k < 4; k++) {
			int index_x = row < 4 ? row - k : 3 - k;
			if(row >= k) {
				x_out.x += queue[index_x].x * d_by[k];
				x_out.y += queue[index_x].y * d_by[k];
				x_out.z += queue[index_x].z * d_by[k];
			} else {
				x_out.x += queue[0].x * d_by[k];
				x_out.y += queue[0].y * d_by[k];		
				x_out.z += queue[0].z * d_by[k];
			}

			int index_y = row <= 4 ? row - k - 1 : 3 - k;
			if(row >= k + 1) {
				x_out.x -= queue_h[index_y].x * d_ay[k];
				x_out.y -= queue_h[index_y].y * d_ay[k];
				x_out.z -= queue_h[index_y].z * d_ay[k];
			} else {
				x_out.x -= queue[0].x * d_ay[k];
				x_out.y -= queue[0].y * d_ay[k];	
				x_out.z -= queue[0].z * d_ay[k];	
			}
		
		//	x_out.x += row >= k ? queue[3 - k].x * d_by[k] : queue[0].x * d_by[k];
		//	x_out.y += row >= k ? queue[3 - k].y * d_by[k] : queue[0].y * d_by[k];
		//	x_out.z += row >= k ? queue[3 - k].z * d_by[k] : queue[0].z * d_by[k];

		//	x_out.x -= row >= k + 1 ? queue_h[3 - k].x * d_ay[k] : queue[0].x * d_ay[k];	
		//	x_out.y -= row >= k + 1 ? queue_h[3 - k].y * d_ay[k] : queue[0].y * d_ay[k];
		//	x_out.z -= row >= k + 1 ? queue_h[3 - k].z * d_ay[k] : queue[0].z * d_ay[k];
		}
		if(row < 4) {
			queue_h[row] = x_out;
		} else {
			advance( queue_h, 3 );
			queue_h[3] = x_out;
		}
		od[processIdx] = cvt_float4_to_uint(x_out);
	}	

	processIdx -= width;
	// backward pass
	for(int row = height - 1; row >= 0; row--, processIdx -= width) { 
		float4 x_out = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		if(row == height - 1) queue[0] = cvt_uint_to_float4(id[processIdx]);
		#pragma unroll 4
		for(int k = 0; k < 4; k++) {
			int index = row >= height - 4 ? height - row - k - 2: 3 - k;
			if(row < height - k - 1) {
				x_out.x += queue[index].x * d_by_[k] - queue_h[index].x * d_ay[k];
				x_out.y += queue[index].y * d_by_[k] - queue_h[index].y * d_ay[k];
				x_out.z += queue[index].z * d_by_[k] - queue_h[index].z * d_ay[k];
			} else {
				x_out.x += queue[0].x * (d_by_[k] - d_ay[k]);
				x_out.y += queue[0].y * (d_by_[k] - d_ay[k]);
				x_out.z += queue[0].z * (d_by_[k] - d_ay[k]);
			}
		
		//	x_out.x += row < height - k - 1 ? queue[3 - k].x * d_by_[k] : queue[0].x * d_by_[k];
		//	x_out.y += row < height - k - 1 ? queue[3 - k].y * d_by_[k] : queue[0].y * d_by_[k];
		//	x_out.z += row < height - k - 1 ? queue[3 - k].z * d_by_[k] : queue[0].z * d_by_[k];
		//	
		//	x_out.x -= row < height - k - 1 ? queue_h[3 - k].x * d_ay[k] : queue[0].x * d_ay[k];
		//	x_out.y -= row < height - k - 1 ? queue_h[3 - k].y * d_ay[k] : queue[0].y * d_ay[k];
		//	x_out.z -= row < height - k - 1 ? queue_h[3 - k].z * d_ay[k] : queue[0].z * d_ay[k];
		}
		if(row >= height - 4) {
			queue_h[height - 1- row] = x_out;
			if(row < height - 1) {
				queue[height - 1 - row] = cvt_uint_to_float4(id[processIdx]);
			}	
		} else {
			advance( queue_h, 3 );
			queue_h[3] = x_out;

			advance( queue, 3 );
			queue[3] = cvt_uint_to_float4(id[processIdx]);
		}
		temp[processIdx] = cvt_float4_to_uint(x_out);
		float4 x_fwd = cvt_uint_to_float4(od[processIdx]);
		x_out.x += x_fwd.x; x_out.y += x_fwd.y; x_out.z += x_fwd.z;	
		od[processIdx] = cvt_float4_to_uint(x_out);
	//	od[processIdx] = id[processIdx];
	}
	}

}

__global__ void recursive_gaussian_rgba_row_fwdpass(uint * id, uint * od, const int width, const int height) 
{
	unsigned int tx = threadIdx.x + blockIdx.x * blockDim.x;

	__shared__ float4 queue_smem[4 * BlockXRecurGaussianRgba];
	__shared__ float4 queue_h_smem[4 * BlockXRecurGaussianRgba];
	if(tx < width) {
	od += tx;
	id += tx;

	float4 * queue = queue_smem + threadIdx.x;
	float4 * queue_h = queue_h_smem + threadIdx.x;
//	float4 queue[4];
//	float4 queue_h[4];
	
	// forward pass
	int processIdx = 0;
	for(int row = 0; row < height; row++, processIdx += width) {
		float4 x_out = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		if(row < 4) {
			queue[row * blockDim.x] = cvt_uint_to_float4(id[processIdx]);
	//		queue[row] = cvt_uint_to_float4(id[processIdx]);
		} else {
			advance_shared( queue, 3 );
	//		advance( queue, 3 );
			queue[3 * blockDim.x] = cvt_uint_to_float4(id[processIdx]);
	//		queue[3] = cvt_uint_to_float4(id[processIdx]);
		}
		#pragma unroll 4
		for(int k = 0; k < 4; k++) {
			int index_x = row < 4 ? row - k : 3 - k;
			if(row >= k) {
				x_out.x += queue[index_x * blockDim.x].x * d_by[k];
				x_out.y += queue[index_x * blockDim.x].y * d_by[k];
				x_out.z += queue[index_x * blockDim.x].z * d_by[k];
		//		x_out.x += queue[index_x].x * d_by[k];
		//		x_out.y += queue[index_x].y * d_by[k];
		//		x_out.z += queue[index_x].z * d_by[k];
			} else {
				x_out.x += queue[0].x * d_by[k];
				x_out.y += queue[0].y * d_by[k];		
				x_out.z += queue[0].z * d_by[k];
			}

			int index_y = row <= 4 ? row - k - 1 : 3 - k;
			if(row >= k + 1) {
				x_out.x -= queue_h[index_y * blockDim.x].x * d_ay[k];
				x_out.y -= queue_h[index_y * blockDim.x].y * d_ay[k];
				x_out.z -= queue_h[index_y * blockDim.x].z * d_ay[k];
		//		x_out.x -= queue_h[index_y].x * d_ay[k]; 		
		//		x_out.y -= queue_h[index_y].y * d_ay[k];
		//		x_out.z -= queue_h[index_y].z * d_ay[k]; 	
			} else {
				x_out.x -= queue[0].x * d_ay[k];
				x_out.y -= queue[0].y * d_ay[k];	
				x_out.z -= queue[0].z * d_ay[k];	
			}
		
		//	x_out.x += row >= k ? queue[3 - k].x * d_by[k] : queue[0].x * d_by[k];
		//	x_out.y += row >= k ? queue[3 - k].y * d_by[k] : queue[0].y * d_by[k];
		//	x_out.z += row >= k ? queue[3 - k].z * d_by[k] : queue[0].z * d_by[k];

		//	x_out.x -= row >= k + 1 ? queue_h[3 - k].x * d_ay[k] : queue[0].x * d_ay[k];	
		//	x_out.y -= row >= k + 1 ? queue_h[3 - k].y * d_ay[k] : queue[0].y * d_ay[k];
		//	x_out.z -= row >= k + 1 ? queue_h[3 - k].z * d_ay[k] : queue[0].z * d_ay[k];
		}
		if(row < 4) {
			queue_h[row * blockDim.x] = x_out;
	//		queue_h[row] = x_out;
		} else {
			advance_shared( queue_h, 3 );
	//		advance( queue_h, 3 );
			queue_h[3 * blockDim.x] = x_out;
	//		queue_h[3] = x_out;
		}
		od[processIdx] = cvt_float4_to_uint(x_out);
	}	
	}

}

__global__ void recursive_gaussian_rgba_row_revpass(uint __restrict__ * id, uint __restrict__ * od, const int width, const int height) 
{
	unsigned int tx = threadIdx.x + blockIdx.x * blockDim.x;
//	__shared__ float4 queue_smem[4 * BlockXRecurGaussianRgba];
//	__shared__ float4 queue_h_smem[4 * BlockXRecurGaussianRgba];

	if(tx < width) {
	od += tx;
	id += tx;

//	float4 * queue = queue_smem + threadIdx.x;
//	float4 * queue_h = queue_h_smem + threadIdx.x;

	float4 queue[4];
	float4 queue_h[4];
	
	int processIdx = (height - 1) * width;
	// backward pass
	for(int row = height - 1; row >= 0; row--, processIdx -= width) { 
		float4 x_out = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		if(row == height - 1) queue[0] = cvt_uint_to_float4(id[processIdx]);
		#pragma unroll 4
		for(int k = 0; k < 4; k++) {
			int index = row >= height - 4 ? height - row - k - 2: 3 - k;
			if(row < height - k - 1) {
				x_out.x += queue[index].x * d_by_[k] - queue_h[index].x * d_ay[k];
				x_out.y += queue[index].y * d_by_[k] - queue_h[index].y * d_ay[k];
				x_out.z += queue[index].z * d_by_[k] - queue_h[index].z * d_ay[k];
		//		x_out.x += queue[index * blockDim.x].x * d_by_[k] - queue_h[index * blockDim.x].x * d_ay[k];
		//		x_out.y += queue[index * blockDim.x].y * d_by_[k] - queue_h[index * blockDim.x].y * d_ay[k];
		//		x_out.z += queue[index * blockDim.x].z * d_by_[k] - queue_h[index * blockDim.x].z * d_ay[k];
			} else {
				x_out.x += queue[0].x * (d_by_[k] - d_ay[k]);
				x_out.y += queue[0].y * (d_by_[k] - d_ay[k]);
				x_out.z += queue[0].z * (d_by_[k] - d_ay[k]);
			}
		
		//	x_out.x += row < height - k - 1 ? queue[3 - k].x * d_by_[k] : queue[0].x * d_by_[k];
		//	x_out.y += row < height - k - 1 ? queue[3 - k].y * d_by_[k] : queue[0].y * d_by_[k];
		//	x_out.z += row < height - k - 1 ? queue[3 - k].z * d_by_[k] : queue[0].z * d_by_[k];
		//	
		//	x_out.x -= row < height - k - 1 ? queue_h[3 - k].x * d_ay[k] : queue[0].x * d_ay[k];
		//	x_out.y -= row < height - k - 1 ? queue_h[3 - k].y * d_ay[k] : queue[0].y * d_ay[k];
		//	x_out.z -= row < height - k - 1 ? queue_h[3 - k].z * d_ay[k] : queue[0].z * d_ay[k];
		}
		if(row >= height - 4) {
			queue_h[height - 1- row] = x_out;
		//	queue_h[(height - row) * blockDim.x - blockDim.x] = x_out;
			if(row < height - 1) {
				queue[height - 1 - row] = cvt_uint_to_float4(id[processIdx]);
		//		queue[(height - row) * blockDim.x - blockDim.x] = x_out;
			}	
		} else {
			advance( queue_h, 3 );
		//	advance_shared( queue_h, 3 );
			queue_h[3] = x_out;
		//	queue_h[3 * blockDim.x] = x_out;

			advance( queue, 3 );
		//	advance_shared( queue, 3 );
			queue[3] = cvt_uint_to_float4(id[processIdx]);
		//	queue[3 * blockDim.x] = cvt_uint_to_float4(id[processIdx]);
		}
		od[processIdx] = cvt_float4_to_uint(x_out);
	}
	}

}

__global__ void box_filter(uint * id, uint * od, const int width, const int height, const int radius, const float scale) 
{
	unsigned int tx = threadIdx.x + blockIdx.x * blockDim.x;

	if(tx < width) {
	od += tx;
	id += tx;
	
	int processIdx = 0;
	float4 tempRow = cvt_uint_to_float4(id[processIdx]);
	float4 sumArea = tempRow;
	sumArea.x *= radius + 1;
	sumArea.y *= radius + 1;
	sumArea.z *= radius + 1;
	
	for(int r = 1; r <= radius; r++) {
		float4 temp = cvt_uint_to_float4(id[r * width]);
		sumArea.x += temp.x; sumArea.y += temp.y; sumArea.z += temp.z;
	}

	float4 val;
	val.x = sumArea.x * scale;
	val.y = sumArea.y * scale;
	val.z = sumArea.z * scale;

	od[processIdx] = cvt_float4_to_uint(val);

	processIdx += width;
	for(int row = 1; row <= radius; row++, processIdx += width) {
		float4 temp = cvt_uint_to_float4(id[processIdx + radius * width]);
		sumArea.x += temp.x; sumArea.y += temp.y; sumArea.z += temp.z;
		sumArea.x -= tempRow.x; sumArea.y -= tempRow.y; sumArea.z -= tempRow.z;
		val.x = sumArea.x * scale;
		val.y = sumArea.y * scale;
		val.z = sumArea.z * scale;
		od[processIdx] = cvt_float4_to_uint(val);
	}	

	for(int row = radius + 1; row < height - radius; row++, processIdx += width) {
		float4 temp = cvt_uint_to_float4(id[processIdx + radius * width]);
		sumArea.x += temp.x; sumArea.y += temp.y; sumArea.z += temp.z;
		temp = cvt_uint_to_float4(id[processIdx - (radius + 1) * width]);
		sumArea.x -= temp.x; sumArea.y -= temp.y; sumArea.z -= temp.z;
		val.x = sumArea.x * scale;
		val.y = sumArea.y * scale;
		val.z = sumArea.z * scale;
		od[processIdx] = cvt_float4_to_uint(val);
	}

	tempRow = cvt_uint_to_float4(id[(height - 1) * width]);

	for(int row = height - radius; row < height; row++, processIdx += width) {
		sumArea.x += tempRow.x; sumArea.y += tempRow.y; sumArea.z += tempRow.z;
		float4 temp = cvt_uint_to_float4(id[processIdx - (radius + 1) * width]);
		sumArea.x -= temp.x; sumArea.y -= temp.y; sumArea.z -= temp.z;
		val.x = sumArea.x * scale;
		val.y = sumArea.y * scale;
		val.z = sumArea.z * scale;
		od[processIdx] = cvt_float4_to_uint(val);
	}
	}
}

extern "C" void recursive_gaussian_gpu(Mat& src, Mat& dst, double sigmaX, double sigmaY)
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

	long int totalBytes = 2 * ibytes + 3 * sizeof(uint) * width32 * height32;
	uchar * d_data;
	device_memory_allocate((void**)&d_data, totalBytes);
	uchar * d_src = d_data;
	uchar * d_dst = d_src + ibytes;
	uint * d_rgba = (uint*)(d_dst + ibytes);
	uint * d_temp = d_rgba + width32 * height32;
	uint * d_temp_1 = d_temp + width32 * height32;

	uchar * h_src = src.ptr<uchar>(0);
	copy_to_device(d_src, h_src, ibytes);

	dim3 gridCvtUchar3toRgba(i_div_up(width32 * Channel3, BlockXcvtUchar3toRgba), i_div_up(height32, BlockYcvtUchar3toRgba), 1);
	dim3 blockCvtUchar3toRgba(BlockXcvtUchar3toRgba, BlockYcvtUchar3toRgba, 1);

	dim3 gridTransp_r2h(i_div_up(width32, BlockXTransp), i_div_up(height32, BlockYTransp), 1);
	dim3 gridTransp_h2r(i_div_up(height32, BlockXTransp), i_div_up(width32, BlockYTransp), 1);
	dim3 blockTransp(BlockXTransp, BlockYTransp, 1);

	prepare_coeffs_recursive_gaussian_x(sigmaX);
	prepare_coeffs_recursive_gaussian_y(sigmaY);

	// warm up
	float gputime = 0.0;

	cudaEvent_t start, end;
	create_event(&start);
	create_event(&end);

	record_event(start);
	cudaStream_t stream_fwd;
	cudaStream_t stream_rev;
	stream_create(&stream_fwd);
	stream_create(&stream_rev);

	cvt_uchar3_to_rgba<<<gridCvtUchar3toRgba, blockCvtUchar3toRgba>>>(d_src, d_rgba, width32, height32);
	check_cuda_error("kernel cvt_uchar3_to_rgba");
	// process col
//	recursive_gaussian_rgba_col<<<i_div_up(width32, BlockXRecurGaussianRgba), BlockXRecurGaussianRgba>>>(d_rgba, d_temp, d_temp_1, width32, height32);
	cudaStreamSynchronize(0);
	recursive_gaussian_rgba_col_fwdpass<<<i_div_up(width32, BlockXRecurGaussianRgba), BlockXRecurGaussianRgba, 0, stream_fwd>>>(d_rgba, d_temp, width32, height32);
	recursive_gaussian_rgba_col_revpass<<<i_div_up(width32, BlockXRecurGaussianRgba), BlockXRecurGaussianRgba, 0, stream_rev>>>(d_rgba, d_temp_1, width32, height32);
	cudaStreamSynchronize(stream_fwd);
	cudaStreamSynchronize(stream_rev);
	recursive_gaussian_rgba_sum<<<gridTransp_r2h, blockTransp>>>(d_temp_1, d_temp, width32, height32);
	check_cuda_error("kernel recursive_gaussian_rgba_row");
//	box_filter<<<i_div_up(width32, BlockXRecurGaussianRgba), BlockXRecurGaussianRgba>>>(d_rgba, d_temp, width32, height32, sigma, 1.0f / (2.0f * sigma + 1.0f));
//	check_cuda_error("kernel box_filter");
	// transpose
	transpose<<<gridTransp_r2h, blockTransp>>>(d_temp, d_rgba, width32, height32);
	check_cuda_error("kernel transpose");
	cudaStreamSynchronize(0);
	// process row			
//	recursive_gaussian_rgba_row<<<i_div_up(height32, BlockXRecurGaussianRgba), BlockXRecurGaussianRgba>>>(d_rgba, d_temp, d_temp_1, height32, width32);
	recursive_gaussian_rgba_row_fwdpass<<<i_div_up(height32, BlockXRecurGaussianRgba), BlockXRecurGaussianRgba, 0, stream_fwd>>>(d_rgba, d_temp, height32, width32);
	recursive_gaussian_rgba_row_revpass<<<i_div_up(height32, BlockXRecurGaussianRgba), BlockXRecurGaussianRgba, 0, stream_rev>>>(d_rgba, d_temp_1, height32, width32);
	cudaStreamSynchronize(stream_fwd);
	cudaStreamSynchronize(stream_rev);
	recursive_gaussian_rgba_sum<<<gridTransp_h2r, blockTransp>>>(d_temp_1, d_temp, height32, width32);
	check_cuda_error("kernel recursive_gaussian_rgba");
//	box_filter<<<i_div_up(height32, BlockXRecurGaussianRgba), BlockXRecurGaussianRgba>>>(d_rgba, d_temp, height32, width32, sigma, 1.0f / (2.0f * sigma + 1.0f));
//	check_cuda_error("kernel box_filter");
	// transpose
	transpose<<<gridTransp_h2r, blockTransp>>>(d_temp, d_rgba, height32, width32);
	check_cuda_error("kernel transpose");
	
	cvt_rgba_to_uchar3<<<gridCvtUchar3toRgba, blockCvtUchar3toRgba>>>(d_rgba, d_dst, width32, height32);
	check_cuda_error("kernel cvt_rgba_to_uchar3");
	
	record_event(end);
	sync_event(end);
	event_elapsed_time(&gputime, start, end);	
	destroy_event(start);
	destroy_event(end);
	fprintf(stdout, "INFO: elapsed time of cuda kernel cvt_uchar3_to_rgba() and cvt_rgba_to_uchar3() is %f\n", gputime * 0.001);	
	
	stream_destroy(stream_fwd);
	stream_destroy(stream_rev);

	uchar * h_dst = dst.ptr<uchar>(0);
	copy_to_host(h_dst, d_dst, ibytes);
	resize(dst, dst, Size(width, height));
	free_device_memory(d_data);
}

class CUDABilateralFilter {
public:
	CUDABilateralFilter(double factor_, double spatialSimga_x_, double spatialSimga_y_, unsigned int spatialRadius_x_, unsigned int spatialRadius_y_, double rangeParams_) : factor(factor_), spatialSimga_x(spatialSimga_x_), spatialSimga_y(spatialSimga_y_), spatialRadius_x(spatialRadius_x_), spatialRadius_y(spatialRadius_y_), rangeParams(rangeParams_) { }; 
	~CUDABilateralFilter() { };
	void apply(Mat& src, Mat& dst);
public:
	double factor;
	double spatialSimga_x;
	double spatialSimga_y;
	unsigned int spatialRadius_x;
	unsigned int spatialRadius_y;
	double rangeParams;
	static const unsigned int MaxColorDepth;
	static const double Epsilon;
};

const unsigned int CUDABilateralFilter::MaxColorDepth = 0xff;
const double CUDABilateralFilter::Epsilon = 1e-14;

struct recursive_gaussian_params_t {
	cudaStream_t * stream_default;
	cudaStream_t * stream_fwd;
	cudaStream_t * stream_rev;
	uint * d_in;
	uint * d_out;
	uint * d_mem_var;
	unsigned int width32;
	unsigned int height32;
	recursive_gaussian_params_t(cudaStream_t * stream_default_, cudaStream_t * stream_fwd_, cudaStream_t * stream_rev_, uint * d_in_, uint * d_out_, uint * d_mem_var_, uint width32_, uint height32_) : stream_default(stream_default_), stream_fwd(stream_fwd_), stream_rev(stream_rev_), d_in(d_in_), d_out(d_out_), d_mem_var(d_mem_var_), width32(width32_), height32(height32_) { };
};

void* recursive_gaussian_thread(void * ptr)
{
	recursive_gaussian_params_t * params = (recursive_gaussian_params_t*)ptr;
	cudaStream_t * stream_default = params->stream_default;
	cudaStream_t * stream_fwd     = params->stream_fwd;
	cudaStream_t * stream_rev     = params->stream_rev;	
	uint * d_in = params->d_in;
	uint * d_out = params->d_out;
	uint * d_mem_var = params->d_mem_var;
	uint width32 = params->width32;
	uint height32 = params->height32;

	int spatialSimga_x = 10;
	int spatialSimga_y = 10;

	dim3 gridTransp_r2h(i_div_up(width32, BlockXTransp), i_div_up(height32, BlockYTransp), 1);
	dim3 gridTransp_h2r(i_div_up(height32, BlockXTransp), i_div_up(width32, BlockYTransp), 1);
	dim3 blockTransp(BlockXTransp, BlockYTransp, 1);

//	recursive_gaussian_rgba_col_fwdpass<<<i_div_up(width32, BlockXRecurGaussianRgba), BlockXRecurGaussianRgba, 0, *stream_fwd>>>(d_in, d_out, width32, height32);
//	recursive_gaussian_rgba_col_revpass<<<i_div_up(width32, BlockXRecurGaussianRgba), BlockXRecurGaussianRgba, 0, *stream_rev>>>(d_in, d_mem_var, width32, height32);
//	cudaStreamSynchronize(*stream_fwd);
//	cudaStreamSynchronize(*stream_rev);	
//	recursive_gaussian_rgba_sum<<<gridTransp_r2h, blockTransp, 0, *stream_default>>>(d_out, d_mem_var, width32, height32);

	box_filter<<<i_div_up(width32, BlockXRecurGaussianRgba), BlockXRecurGaussianRgba, 0, *stream_default>>>(d_in, d_out, width32, height32, spatialSimga_y, 1.0f / (2.0f * spatialSimga_y + 1.0f));

	check_cuda_error("kernel recursive_gaussian_rgba_col");
	// transpose
	transpose<<<gridTransp_r2h, blockTransp, 0, *stream_default>>>(d_out, d_in, width32, height32);
	check_cuda_error("kernel transpose");
	cudaStreamSynchronize(*stream_default);
	
//	recursive_gaussian_rgba_row_fwdpass<<<i_div_up(height32, BlockXRecurGaussianRgba), BlockXRecurGaussianRgba, 0, *stream_fwd>>>(d_in, d_out, height32, width32);
//	recursive_gaussian_rgba_row_revpass<<<i_div_up(height32, BlockXRecurGaussianRgba), BlockXRecurGaussianRgba, 0, *stream_rev>>>(d_in, d_mem_var, height32, width32);
//	cudaStreamSynchronize(*stream_fwd);
//	cudaStreamSynchronize(*stream_rev);
//	recursive_gaussian_rgba_sum<<<gridTransp_h2r, blockTransp, 0, *stream_default>>>(d_out, d_mem_var, height32, width32);

	box_filter<<<i_div_up(height32, BlockXRecurGaussianRgba), BlockXRecurGaussianRgba, 0, *stream_default>>>(d_in, d_out, height32, width32, spatialSimga_x, 1.0f / (2.0f * spatialSimga_x + 1.0f));

	check_cuda_error("kernel recursive_gaussian_rgba_row");
	transpose<<<gridTransp_h2r, blockTransp, 0, *stream_default>>>(d_out, d_in, height32, width32);
	cudaStreamSynchronize(*stream_default);

	pthread_exit(NULL);
}

__global__ void compute_wk_jk(uint * d_in, uint * d_wk, uint * d_jk, const int width, const int height, const double k, const double twosqrangei)
{
	unsigned int tx = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int ty = threadIdx.y + blockIdx.y * blockDim.y;

	if(tx < width && ty < height) {
		unsigned int inputIdx = ty * width + tx;
	
		float4 I = cvt_uint_to_float4(d_in[inputIdx]);
		float4 rgba;
//		float factor = expf(- ((I.x - k) * (I.x - k) + (I.y - k) * (I.y - k) + (I.z - k) * (I.z - k)) * twosqrangei);
//		float4 rgba = make_float4(factor, factor, factor, factor);	

		rgba.x = __expf(- (I.x - k) * (I.x - k) * twosqrangei);
		rgba.y = __expf(- (I.y - k) * (I.y - k) * twosqrangei);
		rgba.z = __expf(- (I.z - k) * (I.z - k) * twosqrangei);
		d_wk[inputIdx] = cvt_float4_to_uint(rgba);
		
		rgba.x *= I.x;
		rgba.y *= I.y;
		rgba.z *= I.z;

		d_jk[inputIdx] = cvt_float4_to_uint(rgba);
	}
}

__global__ void compute_jbk(uint * d_jk_, uint * d_wk, uint * d_jk, const int width, const int height)
{
	unsigned int tx = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int ty = threadIdx.y + blockIdx.y * blockDim.y;

	if(tx < width && ty < height) {
		unsigned int inputIdx = ty * width + tx;
	
		float4 wk = cvt_uint_to_float4(d_wk[inputIdx]);
		float4 jk = cvt_uint_to_float4(d_jk[inputIdx]);
		
	//	wk.x += 1e-4; wk.y += 1e-4; wk.z += 1e-4;
		jk.x = jk.x / wk.x; jk.y = jk.y / wk.y; jk.z = jk.z / wk.z;
		d_jk_[inputIdx] = cvt_float4_to_uint(jk);
	}
}

__global__ void bilateral_filter_lininterp(uint * d_out, uint * d_in, uint * d_wk, uint * d_jk, uint * d_jk_, const int width, const int height, const double k, const double k_next)
{
	unsigned int tx = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int ty = threadIdx.y + blockIdx.y * blockDim.y;
	double stepi = 1.0f / (k_next - k);

	if(tx < width && ty < height) {
		unsigned int inputIdx = ty * width + tx;
	
		float4 I = cvt_uint_to_float4(d_in[inputIdx]);
		float4 wk = cvt_uint_to_float4(d_wk[inputIdx]);
		float4 jk = cvt_uint_to_float4(d_jk[inputIdx]);
		float4 jk_ = cvt_uint_to_float4(d_jk_[inputIdx]);

	//	wk.x += 1e-4; wk.y += 1e-4; wk.z += 1e-4;
		jk.x = jk.x / wk.x; jk.y = jk.y / wk.y; jk.z = jk.z / wk.z;
		d_jk_[inputIdx] = cvt_float4_to_uint(jk);

		int predx = I.x >= k && I.x < k_next;
		int predy = I.y >= k && I.y < k_next; 
		int predz = I.z >= k && I.z < k_next;
		float4 out = cvt_uint_to_float4(d_out[inputIdx]);
		float weightx = stepi * (k_next - I.x);
		float weighty = stepi * (k_next - I.y);
		float weightz = stepi * (k_next - I.z);
		out.x = predx ? jk_.x * weightx + jk.x * (1.f - weightx) : out.x;
		out.y = predy ? jk_.y * weighty + jk.y * (1.f - weighty) : out.y;
		out.z = predz ? jk_.z * weightz + jk.z * (1.f - weightz) : out.z;
	//	out.x = predx ? stepi * (jk_.x * (k_next - I.x) + jk.x * (I.x - k)) : out.x;
	//	out.y = predy ? stepi * (jk_.y * (k_next - I.y) + jk.y * (I.y - k)) : out.y;
	//	out.z = predz ? stepi * (jk_.z * (k_next - I.z) + jk.z * (I.z - k)) : out.z;
		d_out[inputIdx] = cvt_float4_to_uint(out);
	}
}

void CUDABilateralFilter::apply(Mat& src, Mat& dst) 
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
	double twosqrangei = 1.0f / (2.0f * rangeParams * rangeParams);
	const double step = factor / MaxColorDepth;
	const int nDepth = std::abs((int)(1.0 / step) - (1.0 / step)) < Epsilon ? (int)(1.0 / step) + 1 : (int)(1.0 / step) + 2;

	long int totalBytes = 2 * ibytes;
	uchar * d_data;
	device_memory_allocate((void**)&d_data, totalBytes);
	uchar * d_src = d_data; uchar * d_dst = d_data + ibytes;
	
	long int totalMemVariabBytes = 9 * sizeof(float) * width32 * height32;
	uint * d_mem_var;
	device_memory_allocate((void**)&d_mem_var, totalMemVariabBytes);
	long int wh = width32 * height32;
	uint * d_rgba = d_mem_var;
	uint * d_wk = d_mem_var + wh;
	uint * d_wk_ = d_wk + wh;
	uint * d_jk = d_wk_ + wh;
	uint * d_jk_ = d_jk + wh;
	uint * d_jk_p = d_jk_ + wh;
	uint * d_mem_var_1 = d_jk_p + wh;
	uint * d_mem_var_2 = d_mem_var_1 + wh;
	uint * d_out = d_mem_var_2 + wh;

//	cuda_memset(d_out, wh * sizeof(uint), 0);	

	uchar * h_src = src.ptr<uchar>(0);
	copy_to_device(d_src, h_src, ibytes);
	
	dim3 gridCvtUchar3toRgba(i_div_up(width32 * Channel3, BlockXcvtUchar3toRgba), i_div_up(height32, BlockYcvtUchar3toRgba), 1);
	dim3 blockCvtUchar3toRgba(BlockXcvtUchar3toRgba, BlockYcvtUchar3toRgba, 1);

	dim3 gridTransp_r2h(i_div_up(width32, BlockXTransp), i_div_up(height32, BlockYTransp), 1);
	dim3 gridTransp_h2r(i_div_up(height32, BlockXTransp), i_div_up(width32, BlockYTransp), 1);
	dim3 blockTransp(BlockXTransp, BlockYTransp, 1);

	prepare_coeffs_recursive_gaussian_x(spatialSimga_x);
	prepare_coeffs_recursive_gaussian_y(spatialSimga_y);

	cudaStream_t stream_default_wk;
	cudaStream_t stream_fwd_wk;
	cudaStream_t stream_rev_wk;
	
	cudaStream_t stream_default_jk;
	cudaStream_t stream_fwd_jk;
	cudaStream_t stream_rev_jk;

	stream_create(&stream_default_wk);
	stream_create(&stream_fwd_wk);
	stream_create(&stream_rev_wk);

	stream_create(&stream_default_jk);
	stream_create(&stream_fwd_jk);
	stream_create(&stream_rev_jk);

	float gputime = 0.0;

	cudaEvent_t start, end;
	create_event(&start);
	create_event(&end);

	record_event(start);

	cvt_uchar3_to_rgba<<<gridCvtUchar3toRgba, blockCvtUchar3toRgba>>>(d_src, d_rgba, width32, height32);

//	compute_wk_jk<<<gridTransp_r2h, blockTransp>>>(d_rgba, d_wk, d_jk, width32, height32, 0.0, twosqrangei); 
//
//	cudaStreamSynchronize(0);
//
	pthread_t thrd_wk, thrd_jk;
	recursive_gaussian_params_t params_thrd_wk(&stream_default_wk, &stream_fwd_wk, &stream_rev_wk, d_wk, d_wk_, d_mem_var_1, width32, height32);
	recursive_gaussian_params_t params_thrd_jk(&stream_default_jk, &stream_fwd_jk, &stream_rev_jk, d_jk, d_jk_, d_mem_var_2, width32, height32);	
	pthread_attr_t attr;
	void * status;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
		
	int ret;
//	ret = pthread_create(&thrd_wk, &attr, recursive_gaussian_thread, (void*)(&params_thrd_wk));
//	if(ret) fprintf(stderr, "ERROR: return code from pthread_create() is %d", ret);
//	ret = pthread_create(&thrd_jk, &attr, recursive_gaussian_thread, (void*)(&params_thrd_jk));
//	if(ret) fprintf(stderr, "ERROR: return code from pthread_create() is %d", ret);
//	pthread_join(thrd_wk, &status);
//	if(ret) fprintf(stderr, "ERROR: return code from pthread_join() is %d",ret);
//	pthread_join(thrd_jk, &status);
//	if(ret) fprintf(stderr, "ERROR: return code from pthread_join() is %d",ret);
//
//	compute_jbk<<<gridTransp_r2h, blockTransp>>>(d_jk_p, d_jk, d_wk, width32, height32);

	for(int idepth = 0; idepth < nDepth; idepth++) {
		double k = idepth * step;
		compute_wk_jk<<<gridTransp_r2h, blockTransp>>>(d_rgba, d_wk, d_jk, width32, height32, k, twosqrangei); 
		cudaStreamSynchronize(0);

		ret = pthread_create(&thrd_wk, &attr, recursive_gaussian_thread, (void*)(&params_thrd_wk));
		if(ret) fprintf(stderr, "ERROR: return code from pthread_create() is %d", ret);
		ret = pthread_create(&thrd_jk, &attr, recursive_gaussian_thread, (void*)(&params_thrd_jk));
		if(ret) fprintf(stderr, "ERROR: return code from pthread_create() is %d", ret);
		pthread_join(thrd_wk, &status);
		if(ret) fprintf(stderr, "ERROR: return code from pthread_join() is %d",ret);
		pthread_join(thrd_jk, &status);
		if(ret) fprintf(stderr, "ERROR: return code from pthread_join() is %d",ret);
	
		bilateral_filter_lininterp<<<gridTransp_r2h, blockTransp>>>(d_out, d_rgba, d_wk, d_jk, d_jk_p, width32, height32, (idepth - 1) * step, k);

	}
	cvt_rgba_to_uchar3<<<gridCvtUchar3toRgba, blockCvtUchar3toRgba>>>(d_out, d_dst, width32, height32);
	check_cuda_error("kernel cvt_rgba_to_uchar3");
	
	record_event(end);
	sync_event(end);
	event_elapsed_time(&gputime, start, end);	
	destroy_event(start);
	destroy_event(end);
	fprintf(stdout, "INFO: elapsed time of cuda version bilateral filter is %f\n", gputime * 0.001);	
	

	uchar * h_dst = dst.ptr<uchar>(0);
	copy_to_host(h_dst, d_dst, ibytes);
	resize(dst, dst, Size(width, height));

	free_device_memory(d_data);
	free_device_memory(d_mem_var);

	stream_destroy(stream_default_wk);
	stream_destroy(stream_fwd_wk);
	stream_destroy(stream_rev_wk);
	
	stream_destroy(stream_default_jk);
	stream_destroy(stream_fwd_jk);
	stream_destroy(stream_rev_jk);
}

int main(int argc, char * argv[])
{
	if(argc < 6) {
		fprintf(stdout, "Usage: inputfile outputfile sigma range factor\n");
		return - 1;	
	}
	const char * inputimage = argv[1];
	const char * outputimage = argv[2];
	float sigma = atof(argv[3]);
	float range = atof(argv[4]);
	float factor = atof(argv[5]);
	Mat src = imread(inputimage);
	if( !src.data) {
		fprintf(stderr, "ERROR: cannot open input image %s!\n", inputimage);
		exit(EXIT_FAILURE);
	}
	get_device_property(0);

	Mat dst(src.size(), src.type());
	CUDABilateralFilter cuBilateralFilter(factor, sigma, sigma, (int)(3 * sigma), (int)(3 * sigma), range);
	cuBilateralFilter.apply(src, dst);
	imwrite(outputimage, dst);

	return 0;
}
