/* @file: bilateral_filter.cpp
 * @brief: reference, Yang Q., Tan K., Ahuja N., Real-Time O(1) Bilateral Filtering, CVPR, 2009
 * @author: xiaocen
 * @date: 2017.02.17
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
#include <smmintrin.h>
#include <nmmintrin.h>
#include <emmintrin.h>	// SSE2
#include <immintrin.h>	// AVX
//#include <intrin.h>

using namespace cv; // all the new API is put into "cv" namespace. Export its content
using namespace std;

#define EPSILON 1e-14
#define RADIUS 4
#define NUM_THREADS 4

template<typename SRC_TYPE, typename DST_TYPE, int channels> void boxFilter_(const Mat& src, Mat& dst, Size ksize)
{
	// get gaussian kernel
	int kx = ksize.width;
	int ky = ksize.height;
	assert(kx % 2 == 1 && ky % 2 == 1);
	int radiusX = kx / 2;
	int radiusY = ky / 2;
	kx = radiusX + 1;
	ky = radiusY + 1;
	float kerY[ky];
	float kerX[kx];
//	kerX[0] *= 2.0f; kerY[0] *= 2.0f;
//	double sum = 0.0;
//	for(int i = 0; i < kx; i++) {
//		sum += i == 0 ? kerX[i] : 2.0 * kerX[i];
//	}
	for(int ix = 0; ix < kx; ix++) kerX[ix] = 1.f / (2.f * kx - 1.f);
	for(int iy = 0; iy < ky; iy++) kerY[iy] = 1.f / (2.f * ky - 1.f);

	int h = src.rows;
	int w = src.cols;
	int w_ = w;
#ifdef _SSE2
	w = w % 16 == 0 ? w : 16 * (w / 16 + 1);
#endif
	int ww = w + 2 * radiusX;
	int ww_ = w_ + 2 * radiusX;
	int hh = h + 2 * radiusY;
	assert(src.rows == dst.rows && src.cols == dst.cols);
	Mat rowBuff;
	Mat colBuff;
	if(channels == 1) {
		rowBuff = Mat(h, ww, CV_8UC1);
		colBuff = Mat(hh, w, CV_32FC1);
	} 
	else if(channels == 2) {
		rowBuff = Mat(h, ww, CV_8UC2);
		colBuff = Mat(hh, w, CV_32FC2);
	}
	else if(channels == 3) {
		rowBuff = Mat(h, ww, CV_8UC3);
		colBuff = Mat(hh, w, CV_32FC3);
	}
	const int threadNum = 4;
	for(int y = 0; y < h; y++) {
		SRC_TYPE * rowPtr = rowBuff.ptr<SRC_TYPE>(y) + radiusX;
		const SRC_TYPE * srcPtr = src.ptr<SRC_TYPE>(y);
		memcpy(rowPtr, srcPtr, sizeof(SRC_TYPE) * w_);
	}
	
	// left & right
	for(int y = 0; y < h; y++) {
		SRC_TYPE * rowPtr = rowBuff.ptr<SRC_TYPE>(y);
		for(int x = 0; x < radiusX; x++) {
			rowPtr[x] = rowPtr[radiusX];
			rowPtr[x + ww_ - radiusX] = rowPtr[ww_ - radiusX - 1];
		}
	}

#ifdef _SSE2
	int cn = channels;
	assert(w % 16 == 0);
	__m128i z = _mm_setzero_si128();
	// apply gaussian filter
	#ifdef _OPENMP
	#pragma omp parallel num_threads(threadNum) shared(rowBuff, colBuff, dst, kerX, kerY)
	#endif
	{
	// row filter
	#ifdef _OPENMP
	#pragma omp for schedule(static, h / threadNum)
	#endif
	for(int y = 0; y < h; y++) {
		SRC_TYPE * rowPtr = rowBuff.ptr<SRC_TYPE>(y) + radiusX;
		DST_TYPE * colPtr = colBuff.ptr<DST_TYPE>(y + radiusY);
		uchar * srcRaw = reinterpret_cast<uchar*>(rowPtr);
		float * dstRaw = reinterpret_cast<float*>(colPtr);
		int x = 0;
		for( ; x < w * cn; x += 16, srcRaw += 16) {
			__m128 f = _mm_load_ss(kerX);
			f = _mm_shuffle_ps(f, f, 0);
			__m128i x0 = _mm_loadu_si128((__m128i*)(srcRaw));
			__m128i x1, x2, x3, x4, y0;
			x1 = _mm_unpackhi_epi8(x0, z);
			x2 = _mm_unpacklo_epi8(x0, z);
			x3 = _mm_unpackhi_epi16(x2, z);
			x4 = _mm_unpacklo_epi16(x2, z);
			x2 = _mm_unpacklo_epi16(x1, z);
			x1 = _mm_unpackhi_epi16(x1, z);
			__m128 s1, s2, s3, s4;
			s1 = _mm_mul_ps(f, _mm_cvtepi32_ps(x1));
			s2 = _mm_mul_ps(f, _mm_cvtepi32_ps(x2));
			s3 = _mm_mul_ps(f, _mm_cvtepi32_ps(x3));
			s4 = _mm_mul_ps(f, _mm_cvtepi32_ps(x4));
			for(int k = 1; k < kx; k++) {
				f = _mm_load_ss(kerX + k);
				f = _mm_shuffle_ps(f, f, 0);
				uchar * shi = srcRaw + k * cn;
				uchar * slo = srcRaw - k * cn;
				x0 = _mm_loadu_si128((__m128i*)(shi));
				y0 = _mm_loadu_si128((__m128i*)(slo));
				x1 = _mm_unpackhi_epi8(x0, z);
				x2 = _mm_unpacklo_epi8(x0, z);
				x3 = _mm_unpackhi_epi8(y0, z);
				x4 = _mm_unpacklo_epi8(y0, z);
				x1 = _mm_add_epi16(x1, x3);
				x2 = _mm_add_epi16(x2, x4);
			
				x3 = _mm_unpackhi_epi16(x2, z);
				x4 = _mm_unpacklo_epi16(x2, z);
				x2 = _mm_unpacklo_epi16(x1, z);
				x1 = _mm_unpackhi_epi16(x1, z);
				s1 = _mm_add_ps(s1, _mm_mul_ps(f, _mm_cvtepi32_ps(x1)));
				s2 = _mm_add_ps(s2, _mm_mul_ps(f, _mm_cvtepi32_ps(x2)));
				s3 = _mm_add_ps(s3, _mm_mul_ps(f, _mm_cvtepi32_ps(x3)));
				s4 = _mm_add_ps(s4, _mm_mul_ps(f, _mm_cvtepi32_ps(x4)));	
			}
			_mm_storeu_ps(dstRaw + x, s4);
			_mm_storeu_ps(dstRaw + x + 4, s3);
			_mm_storeu_ps(dstRaw + x + 8, s2);
			_mm_storeu_ps(dstRaw + x + 12, s1);
		}
	}
	
	if(omp_get_thread_num() == 0) {
		DST_TYPE * topPtr = colBuff.ptr<DST_TYPE>(0);
		DST_TYPE * botPtr = colBuff.ptr<DST_TYPE>(hh - radiusY);
		DST_TYPE * topLin = colBuff.ptr<DST_TYPE>(radiusY);
		DST_TYPE * botLin = colBuff.ptr<DST_TYPE>(hh - radiusY - 1);	
		for(int y = 0; y < radiusY; y++, topPtr += w, botPtr += w) {
			memcpy(topPtr, topLin, sizeof(DST_TYPE) * w);
			memcpy(botPtr, botLin, sizeof(DST_TYPE) * w);
		}
	}

	// column filter
	#ifdef _OPENMP
	#pragma omp for schedule(static, h / threadNum)
	#endif
	for(int y = 0; y < h; y++) {
		DST_TYPE * srcPtr = colBuff.ptr<DST_TYPE>(y + radiusY);
		SRC_TYPE * dstPtr = dst.ptr<SRC_TYPE>(y);
		float * srcRaw = reinterpret_cast<float*>(srcPtr);
		uchar * dstRaw = reinterpret_cast<uchar*>(dstPtr);
		int x = 0;
		for( ; x < w * cn; x += 16, srcRaw += 16) {
			__m128 f = _mm_load_ss(kerY);
			f = _mm_shuffle_ps(f, f, 0);
			__m128 s1, s2, s3, s4;
			__m128 s0;
			s1 = _mm_loadu_ps(srcRaw);
			s2 = _mm_loadu_ps(srcRaw + 4);
			s3 = _mm_loadu_ps(srcRaw + 8);
			s4 = _mm_loadu_ps(srcRaw + 12);
			s1 = _mm_mul_ps(s1, f);
			s2 = _mm_mul_ps(s2, f);
			s3 = _mm_mul_ps(s3, f);
			s4 = _mm_mul_ps(s4, f);
			for(int k = 1; k < ky; k++) {
				f = _mm_load_ss(kerY + k);
				f = _mm_shuffle_ps(f, f, 0);
				s0 = _mm_add_ps(_mm_loadu_ps(srcRaw + k * w * cn), _mm_loadu_ps(srcRaw - k * w * cn));
				s1 = _mm_add_ps(s1, _mm_mul_ps(f, s0));
				s0 = _mm_add_ps(_mm_loadu_ps(srcRaw + 4 + k * w * cn), _mm_loadu_ps(srcRaw + 4 - k * w * cn));
				s2 = _mm_add_ps(s2, _mm_mul_ps(f, s0));
				s0 = _mm_add_ps(_mm_loadu_ps(srcRaw + 8 + k * w * cn), _mm_loadu_ps(srcRaw + 8 - k * w * cn));
				s3 = _mm_add_ps(s3, _mm_mul_ps(f, s0));
				s0 = _mm_add_ps(_mm_loadu_ps(srcRaw + 12 + k * w * cn), _mm_loadu_ps(srcRaw + 12 - k * w * cn));
				s4 = _mm_add_ps(s4, _mm_mul_ps(f, s0));
			}
			__m128i x1 = _mm_cvttps_epi32(s1);
			__m128i x2 = _mm_cvttps_epi32(s2);
			__m128i x3 = _mm_cvttps_epi32(s3);
			__m128i x4 = _mm_cvttps_epi32(s4);
			x1 = _mm_packs_epi32(x1, x2);
			x2 = _mm_packs_epi32(x3, x4);
			x1 = _mm_packus_epi16(x1, x2);
			uchar buff[16] __attribute__((aligned(16)));
			int len = min(16, w_ * cn - x);
			_mm_store_si128((__m128i*)buff, x1);
			if(len > 0) memcpy(dstRaw + x, buff, len);
		}
	}
	}
#else
	// apply gaussian filter
	#ifdef _OPENMP
	#pragma omp parallel num_threads(threadNum) shared(rowBuff, colBuff, dst, kerX, kerY)
	#endif
	{
	// row filter
	#ifdef _OPENMP
	#pragma omp for schedule(static, h / threadNum)
	#endif
	for(int y = 0; y < hh; y++) {
		SRC_TYPE * rowPtr = rowBuff.ptr<SRC_TYPE>(y) + radiusX;
		DST_TYPE * colPtr = colBuff.ptr<DST_TYPE>(y);
		for(int x = 0; x < w; x++) {
			DST_TYPE vec = kerX[0] * rowPtr[x];
			for(int xx = 1; xx < kx; xx++) {
				vec += kerX[xx] * (rowPtr[x + xx] + rowPtr[x - xx]);
			}
			colPtr[x] = vec;
		}
	}

	if(omp_get_thread_num() == 0) {
		DST_TYPE * topPtr = colBuff.ptr<DST_TYPE>(0);
		DST_TYPE * botPtr = colBuff.ptr<DST_TYPE>(hh - radiusY);
		DST_TYPE * topLin = colBuff.ptr<DST_TYPE>(radiusY);
		DST_TYPE * botLin = colBuff.ptr<DST_TYPE>(hh - radiusY - 1);	
		for(int y = 0; y < radiusY; y++, topPtr += w, botPtr += w) {
			memcpy(topPtr, topLin, sizeof(DST_TYPE) * w);
			memcpy(botPtr, botLin, sizeof(DST_TYPE) * w);
		}
	}	

	// column filter
	#ifdef _OPENMP
	#pragma omp for schedule(static, h / threadNum)
	#endif
	for(int y = 0; y < h; y++) {
		DST_TYPE * srcPtr = colBuff.ptr<DST_TYPE>(y + radiusY);
		SRC_TYPE * dstPtr = dst.ptr<SRC_TYPE>(y);
		for(int x = 0; x < w; x++) {
			DST_TYPE vec = kerY[0] * srcPtr[x];		
			for(int yy = 1; yy < ky; yy++) {
				vec += kerY[yy] * (*(srcPtr + yy * w + x) + *(srcPtr - yy * w + x));
			}
			vec = vec < 0 ? 0.0 : vec; vec = vec > 0xff ? 0xff : vec;
			dstPtr[x] = SRC_TYPE(vec);
		}
	}
	}
#endif



}

void myBoxFilter(const Mat& src, Mat& dst, Size ksize)
{
	int channels = src.channels();
	if(channels == 3) {
		boxFilter_<Vec3b, Vec3f, 3>(src, dst, ksize);
	}
	else if(channels == 2) {
		boxFilter_<Vec2b, Vec2f, 2>(src, dst, ksize);
	}
	else if(channels == 1) {
		boxFilter_<uchar, float, 1>(src, dst, ksize);
	}
}

/* FIR */
template<typename SRC_TYPE, typename DST_TYPE, int channels> void gaussianBlur(const Mat& src, Mat& dst, Size ksize, double sigmaX, double sigmaY)
{
	// get gaussian kernel
	int kx = 2 * static_cast<int>(3.0 * sigmaX) + 1;
	int ky = 2 * static_cast<int>(3.0 * sigmaY) + 1;
	kx = min(kx, ksize.width);
	ky = min(ky, ksize.height);
	assert(kx % 2 == 1 && ky % 2 == 1);
	int radiusX = kx / 2;
	int radiusY = ky / 2;
	kx = radiusX + 1;
	ky = radiusY + 1;
	float kerY[ky];
	float kerX[kx];
	assert(sigmaX > 0 && sigmaY > 0);
	double weightX = 1.0 / (sqrt(2.0 * M_PI) * sigmaX);
	double weightY = 1.0 / (sqrt(2.0 * M_PI) * sigmaY);
	double invSqrSigmaX = 1.0 / (2.0 * sigmaX * sigmaX);
	double invSqrSigmaY = 1.0 / (2.0 * sigmaY * sigmaY);
	double sumY = 0.0;
	for(int y = 0; y < ky; y++) {
		kerY[y] = weightY * exp( - y * y * invSqrSigmaY );
		if(y > 0) sumY += 2.0 * kerY[y];
		else sumY += kerY[y];
	}
	for(int y = 0; y < ky; y++) kerY[y] /= sumY;
	double sumX = 0.0; 
	for(int x = 0; x < kx; x++) {
		kerX[x] = weightX * exp( - x * x * invSqrSigmaX );
		if(x > 0) sumX += 2.0 * kerX[x];
		else sumX += kerX[x];
	}
	for(int x = 0; x < kx; x++) kerX[x] /= sumX;
	
	int h = src.rows;
	int w = src.cols;
	int w_ = w;
#ifdef _SSE2
	w = w % 16 == 0 ? w : 16 * (w / 16 + 1);
#endif
	int ww = w + 2 * radiusX;
	int ww_ = w_ + 2 * radiusX;
	int hh = h + 2 * radiusY;
	assert(src.rows == dst.rows && src.cols == dst.cols);
	Mat rowBuff;
	Mat colBuff;
	if(channels == 1) {
		rowBuff = Mat(h, ww, CV_8UC1);
		colBuff = Mat(hh, w, CV_32FC1);
	} 
	else if(channels == 2) {
		rowBuff = Mat(h, ww, CV_8UC2);
		colBuff = Mat(hh, w, CV_32FC2);
	}
	else if(channels == 3) {
		rowBuff = Mat(h, ww, CV_8UC3);
		colBuff = Mat(hh, w, CV_32FC3);
	}
	const int threadNum = 4;
	for(int y = 0; y < h; y++) {
		SRC_TYPE * rowPtr = rowBuff.ptr<SRC_TYPE>(y) + radiusX;
		const SRC_TYPE * srcPtr = src.ptr<SRC_TYPE>(y);
		memcpy(rowPtr, srcPtr, sizeof(SRC_TYPE) * w_);
	}
	
	// left & right
	for(int y = 0; y < h; y++) {
		SRC_TYPE * rowPtr = rowBuff.ptr<SRC_TYPE>(y);
		for(int x = 0; x < radiusX; x++) {
			rowPtr[x] = rowPtr[radiusX];
			rowPtr[x + ww_ - radiusX] = rowPtr[ww_ - radiusX - 1];
		}
	}

#ifdef _SSE2
	int cn = channels;
	assert(w % 16 == 0);
	__m128i z = _mm_setzero_si128();
	// apply gaussian filter
	#ifdef _OPENMP
	#pragma omp parallel num_threads(threadNum) shared(rowBuff, colBuff, dst, kerX, kerY)
	#endif
	{
	// row filter
	#ifdef _OPENMP
	#pragma omp for schedule(static, h / threadNum)
	#endif
	for(int y = 0; y < h; y++) {
		SRC_TYPE * rowPtr = rowBuff.ptr<SRC_TYPE>(y) + radiusX;
		DST_TYPE * colPtr = colBuff.ptr<DST_TYPE>(y + radiusY);
		uchar * srcRaw = reinterpret_cast<uchar*>(rowPtr);
		float * dstRaw = reinterpret_cast<float*>(colPtr);
		int x = 0;
		for( ; x < w * cn; x += 16, srcRaw += 16) {
			__m128 f = _mm_load_ss(kerX);
			f = _mm_shuffle_ps(f, f, 0);
			__m128i x0 = _mm_loadu_si128((__m128i*)(srcRaw));
			__m128i x1, x2, x3, x4, y0;
			x1 = _mm_unpackhi_epi8(x0, z);
			x2 = _mm_unpacklo_epi8(x0, z);
			x3 = _mm_unpackhi_epi16(x2, z);
			x4 = _mm_unpacklo_epi16(x2, z);
			x2 = _mm_unpacklo_epi16(x1, z);
			x1 = _mm_unpackhi_epi16(x1, z);
			__m128 s1, s2, s3, s4;
			s1 = _mm_mul_ps(f, _mm_cvtepi32_ps(x1));
			s2 = _mm_mul_ps(f, _mm_cvtepi32_ps(x2));
			s3 = _mm_mul_ps(f, _mm_cvtepi32_ps(x3));
			s4 = _mm_mul_ps(f, _mm_cvtepi32_ps(x4));
			for(int k = 1; k < kx; k++) {
				f = _mm_load_ss(kerX + k);
				f = _mm_shuffle_ps(f, f, 0);
				uchar * shi = srcRaw + k * cn;
				uchar * slo = srcRaw - k * cn;
				x0 = _mm_loadu_si128((__m128i*)(shi));
				y0 = _mm_loadu_si128((__m128i*)(slo));
				x1 = _mm_unpackhi_epi8(x0, z);
				x2 = _mm_unpacklo_epi8(x0, z);
				x3 = _mm_unpackhi_epi8(y0, z);
				x4 = _mm_unpacklo_epi8(y0, z);
				x1 = _mm_add_epi16(x1, x3);
				x2 = _mm_add_epi16(x2, x4);
			
				x3 = _mm_unpackhi_epi16(x2, z);
				x4 = _mm_unpacklo_epi16(x2, z);
				x2 = _mm_unpacklo_epi16(x1, z);
				x1 = _mm_unpackhi_epi16(x1, z);
				s1 = _mm_add_ps(s1, _mm_mul_ps(f, _mm_cvtepi32_ps(x1)));
				s2 = _mm_add_ps(s2, _mm_mul_ps(f, _mm_cvtepi32_ps(x2)));
				s3 = _mm_add_ps(s3, _mm_mul_ps(f, _mm_cvtepi32_ps(x3)));
				s4 = _mm_add_ps(s4, _mm_mul_ps(f, _mm_cvtepi32_ps(x4)));	
			}
			_mm_storeu_ps(dstRaw + x, s4);
			_mm_storeu_ps(dstRaw + x + 4, s3);
			_mm_storeu_ps(dstRaw + x + 8, s2);
			_mm_storeu_ps(dstRaw + x + 12, s1);
		}
	}
	
	if(omp_get_thread_num() == 0) {
		DST_TYPE * topPtr = colBuff.ptr<DST_TYPE>(0);
		DST_TYPE * botPtr = colBuff.ptr<DST_TYPE>(hh - radiusY);
		DST_TYPE * topLin = colBuff.ptr<DST_TYPE>(radiusY);
		DST_TYPE * botLin = colBuff.ptr<DST_TYPE>(hh - radiusY - 1);	
		for(int y = 0; y < radiusY; y++, topPtr += w, botPtr += w) {
			memcpy(topPtr, topLin, sizeof(DST_TYPE) * w);
			memcpy(botPtr, botLin, sizeof(DST_TYPE) * w);
		}
	}

	// column filter
	#ifdef _OPENMP
	#pragma omp for schedule(static, h / threadNum)
	#endif
	for(int y = 0; y < h; y++) {
		DST_TYPE * srcPtr = colBuff.ptr<DST_TYPE>(y + radiusY);
		SRC_TYPE * dstPtr = dst.ptr<SRC_TYPE>(y);
		float * srcRaw = reinterpret_cast<float*>(srcPtr);
		uchar * dstRaw = reinterpret_cast<uchar*>(dstPtr);
		int x = 0;
		for( ; x < w * cn; x += 16, srcRaw += 16) {
			__m128 f = _mm_load_ss(kerY);
			f = _mm_shuffle_ps(f, f, 0);
			__m128 s1, s2, s3, s4;
			__m128 s0;
			s1 = _mm_loadu_ps(srcRaw);
			s2 = _mm_loadu_ps(srcRaw + 4);
			s3 = _mm_loadu_ps(srcRaw + 8);
			s4 = _mm_loadu_ps(srcRaw + 12);
			s1 = _mm_mul_ps(s1, f);
			s2 = _mm_mul_ps(s2, f);
			s3 = _mm_mul_ps(s3, f);
			s4 = _mm_mul_ps(s4, f);
			for(int k = 1; k < ky; k++) {
				f = _mm_load_ss(kerY + k);
				f = _mm_shuffle_ps(f, f, 0);
				s0 = _mm_add_ps(_mm_loadu_ps(srcRaw + k * w * cn), _mm_loadu_ps(srcRaw - k * w * cn));
				s1 = _mm_add_ps(s1, _mm_mul_ps(f, s0));
				s0 = _mm_add_ps(_mm_loadu_ps(srcRaw + 4 + k * w * cn), _mm_loadu_ps(srcRaw + 4 - k * w * cn));
				s2 = _mm_add_ps(s2, _mm_mul_ps(f, s0));
				s0 = _mm_add_ps(_mm_loadu_ps(srcRaw + 8 + k * w * cn), _mm_loadu_ps(srcRaw + 8 - k * w * cn));
				s3 = _mm_add_ps(s3, _mm_mul_ps(f, s0));
				s0 = _mm_add_ps(_mm_loadu_ps(srcRaw + 12 + k * w * cn), _mm_loadu_ps(srcRaw + 12 - k * w * cn));
				s4 = _mm_add_ps(s4, _mm_mul_ps(f, s0));
			}
			__m128i x1 = _mm_cvttps_epi32(s1);
			__m128i x2 = _mm_cvttps_epi32(s2);
			__m128i x3 = _mm_cvttps_epi32(s3);
			__m128i x4 = _mm_cvttps_epi32(s4);
			x1 = _mm_packs_epi32(x1, x2);
			x2 = _mm_packs_epi32(x3, x4);
			x1 = _mm_packus_epi16(x1, x2);
			uchar buff[16] __attribute__((aligned(16)));
			int len = min(16, w_ * cn - x);
			_mm_store_si128((__m128i*)buff, x1);
			if(len > 0) memcpy(dstRaw + x, buff, len);
		}
	}
	}
#else
	// apply gaussian filter
	#ifdef _OPENMP
	#pragma omp parallel num_threads(threadNum) shared(rowBuff, colBuff, dst, kerX, kerY)
	#endif
	{
	// row filter
	#ifdef _OPENMP
	#pragma omp for schedule(static, h / threadNum)
	#endif
	for(int y = 0; y < hh; y++) {
		SRC_TYPE * rowPtr = rowBuff.ptr<SRC_TYPE>(y) + radiusX;
		DST_TYPE * colPtr = colBuff.ptr<DST_TYPE>(y);
		for(int x = 0; x < w; x++) {
			DST_TYPE vec = kerX[0] * rowPtr[x];
			for(int xx = 1; xx < kx; xx++) {
				vec += kerX[xx] * (rowPtr[x + xx] + rowPtr[x - xx]);
			}
			colPtr[x] = vec;
		}
	}

	if(omp_get_thread_num() == 0) {
		DST_TYPE * topPtr = colBuff.ptr<DST_TYPE>(0);
		DST_TYPE * botPtr = colBuff.ptr<DST_TYPE>(hh - radiusY);
		DST_TYPE * topLin = colBuff.ptr<DST_TYPE>(radiusY);
		DST_TYPE * botLin = colBuff.ptr<DST_TYPE>(hh - radiusY - 1);	
		for(int y = 0; y < radiusY; y++, topPtr += w, botPtr += w) {
			memcpy(topPtr, topLin, sizeof(DST_TYPE) * w);
			memcpy(botPtr, botLin, sizeof(DST_TYPE) * w);
		}
	}	

	// column filter
	#ifdef _OPENMP
	#pragma omp for schedule(static, h / threadNum)
	#endif
	for(int y = 0; y < h; y++) {
		DST_TYPE * srcPtr = colBuff.ptr<DST_TYPE>(y + radiusY);
		SRC_TYPE * dstPtr = dst.ptr<SRC_TYPE>(y);
		for(int x = 0; x < w; x++) {
			DST_TYPE vec = kerY[0] * srcPtr[x];		
			for(int yy = 1; yy < ky; yy++) {
				vec += kerY[yy] * (*(srcPtr + yy * w + x) + *(srcPtr - yy * w + x));
			}
			vec = vec < 0 ? 0.0 : vec; vec = vec > 0xff ? 0xff : vec;
			dstPtr[x] = SRC_TYPE(vec);
		}
	}
	}
#endif



}

void myGaussianBlur(const Mat& src, Mat& dst, Size ksize, double sigmaX, double sigmaY)
{
	int channels = src.channels();
	if(channels == 3) {
		gaussianBlur<Vec3b, Vec3f, 3>(src, dst, ksize, sigmaX, sigmaY);
	}
	else if(channels == 2) {
		gaussianBlur<Vec2b, Vec2f, 2>(src, dst, ksize, sigmaX, sigmaY);
	}
	else if(channels == 1) {
		gaussianBlur<uchar, float, 1>(src, dst, ksize, sigmaX, sigmaY);
	}
}


/* IIR */
class fastGaussianIIR {
public:
	fastGaussianIIR(double sigmaX_, double sigmaY_);
	~fastGaussianIIR() { };
	void apply(const Mat& src, Mat& dst) const;
private:
	template<typename SRC_TYPE, typename DST_TYPE, int cn> void apply_(const Mat& src, Mat& dst) const;
	void getCoefficients();
	inline void getCoefficients_(double sigma, float a[4], float b[4], float b_[4]);
	inline void rowFilterIIR_32f8u(uchar * src, float * dst, int w);
	inline void columnFilterIIR_8u32f(float * src, uchar * dst, int w);
public:
	double sigmaX;
	double sigmaY;
	float ax[4] __attribute__((aligned(16)));
	float bx[4] __attribute__((aligned(16)));
	float bx_[4] __attribute__((aligned(16)));
	float ay[4] __attribute__((aligned(16)));
	float by[4] __attribute__((aligned(16)));
	float by_[4] __attribute__((aligned(16)));
	static const double deriche4[8];
};

const double fastGaussianIIR::deriche4[] = {1.64847058, -0.64905727, 3.59281342, -0.23814081, 0.63750105, 2.01124176, 1.760522, 1.70738557};

fastGaussianIIR::fastGaussianIIR(double sigmaX_, double sigmaY_) : sigmaX(sigmaX_), sigmaY(sigmaY_)
{
	getCoefficients(); 
};


inline void fastGaussianIIR::rowFilterIIR_32f8u(uchar * src, float * dst, int w)
{
	
}

inline void fastGaussianIIR::columnFilterIIR_8u32f(float * src, uchar * dst, int w)
{

}


template<typename SRC_TYPE, typename DST_TYPE, int cn>
void fastGaussianIIR::apply_(const Mat& src, Mat& dst) const
{
	int h = src.rows;
	int w = src.cols;
	int ww = w + 2 * RADIUS;
	assert(src.type() == dst.type() && src.size() == dst.size());
	Mat buff;
	Mat hcol;
	Mat hrow;
	if(cn == 1) {
		buff = Mat(h, w, CV_32FC1);
		hcol = Mat(h, w, CV_32FC1);
		hrow = Mat(NUM_THREADS, ww, CV_32FC1);
	} 
	else if(cn == 2) {
		buff = Mat(h, w, CV_32FC2);
		hcol = Mat(h, w, CV_32FC2);
		hrow = Mat(NUM_THREADS, ww, CV_32FC2);
	}
	else if(cn == 3) {
		buff = Mat(h, w, CV_32FC3);
		hcol = Mat(h, w, CV_32FC3);
		hrow = Mat(NUM_THREADS, ww, CV_32FC3);
	}

	w *= cn;
//	if(w % 16 != 0) {
//		cout << w << "\n";
//	}
	const int thread_block_size = (w >> 6) << 4;
//	int thread_block_size = w;
	float rowbuff[w];
	const uchar * ptr1 = reinterpret_cast<const uchar*>(src.ptr<SRC_TYPE>(0));
	for(int x = 0; x < w; x++) {
		rowbuff[x] = ptr1[x];
	}
	memset(buff.ptr<uchar>(0), 0, h * w);
	__m128i z = _mm_setzero_si128();
	double t = omp_get_wtime();
	#ifdef _OPENMP
	#pragma omp parallel num_threads(NUM_THREADS) shared(src, dst, buff, hrow, hcol, rowbuff)
	#endif
	{
	#ifdef _OPENMP
	#pragma omp for schedule(dynamic)
	#endif
	for(int x = 0; x < w; x += thread_block_size) { 
		for(int y = 0; y < h; y++) {
			const uchar * src_ptr = reinterpret_cast<const uchar*>(src.ptr<SRC_TYPE>(y)) + x;
			const uchar * row_ptr = reinterpret_cast<const uchar*>(src.ptr<SRC_TYPE>(0)) + x; 
			float * dst_ptr = reinterpret_cast<float*>(buff.ptr<DST_TYPE>(y)) + x;
			float * dst_row_buff = rowbuff + x;
			int ix = 0;
			for(; ix < thread_block_size && x + ix <= w - 16; ix += 16, src_ptr += 16, row_ptr += 16, dst_ptr += 16, dst_row_buff += 16) {
				__m128 s1, s2, s3, s4;
				s1 = _mm_setzero_ps(); s2 = s1; s3= s1; s4 = s1;
				for(int k = 0; k < RADIUS; k++) {
					const uchar * xk_ptr = y >= k ? src_ptr - k * w : row_ptr; 
					__m128 f = _mm_load_ss(by + k);
					f = _mm_shuffle_ps(f, f, 0);
					__m128i x0, x1, x2, x3, x4;
					x0 = _mm_loadu_si128((__m128i*)(xk_ptr));
					x1 = _mm_unpackhi_epi8(x0, z);
					x2 = _mm_unpacklo_epi8(x0, z);
					x3 = _mm_unpackhi_epi16(x2, z);
					x4 = _mm_unpacklo_epi16(x2, z);
					x2 = _mm_unpacklo_epi16(x1, z);
					x1 = _mm_unpackhi_epi16(x1, z);
					s1 = _mm_add_ps(s1, _mm_mul_ps(f, _mm_cvtepi32_ps(x1)));
					s2 = _mm_add_ps(s2, _mm_mul_ps(f, _mm_cvtepi32_ps(x2)));
					s3 = _mm_add_ps(s3, _mm_mul_ps(f, _mm_cvtepi32_ps(x3)));
					s4 = _mm_add_ps(s4, _mm_mul_ps(f, _mm_cvtepi32_ps(x4)));
					float * hk_ptr = y >= k + 1 ? dst_ptr - (k + 1) * w : dst_row_buff;
					f = _mm_load_ss(ay + k);
					f = _mm_shuffle_ps(f, f, 0);
					__m128 y1 = _mm_loadu_ps(hk_ptr + 12);
					__m128 y2 = _mm_loadu_ps(hk_ptr + 8);
					__m128 y3 = _mm_loadu_ps(hk_ptr + 4);
					__m128 y4 = _mm_loadu_ps(hk_ptr);
					s1 = _mm_sub_ps(s1, _mm_mul_ps(f, y1));
					s2 = _mm_sub_ps(s2, _mm_mul_ps(f, y2));
					s3 = _mm_sub_ps(s3, _mm_mul_ps(f, y3));
					s4 = _mm_sub_ps(s4, _mm_mul_ps(f, y4));
				}
				_mm_storeu_ps(dst_ptr, s4);
				_mm_storeu_ps(dst_ptr + 4, s3);
				_mm_storeu_ps(dst_ptr + 8, s2);
				_mm_storeu_ps(dst_ptr + 12, s1);
			}
			for(; ix < thread_block_size && x + ix < w; ix++, src_ptr++, row_ptr++, dst_ptr++) {
				for(int k = 0; k < RADIUS; k++) {
					if(y >= k) {
						*dst_ptr += by[k] * (*(src_ptr - k * w));
					} else {
						*dst_ptr += by[k] * (*row_ptr);
					}
					if(y >= k + 1) {
						*dst_ptr -= ay[k] * (*(dst_ptr - (k + 1) * w));
					} else {
						*dst_ptr -= ay[k] * (*row_ptr);
					}
				}
			}
		}
		for(int y = h - 1; y >= 0; y--) {
			const uchar * src_ptr = reinterpret_cast<const uchar*>(src.ptr<SRC_TYPE>(y)) + x;
			const uchar * row_ptr = reinterpret_cast<const uchar*>(src.ptr<SRC_TYPE>(h - 1)) + x; 
			float * dst_ptr = reinterpret_cast<float*>(hcol.ptr<DST_TYPE>(y)) + x;
			float * buff_ptr = reinterpret_cast<float*>(buff.ptr<DST_TYPE>(y)) + x;
			int ix = 0;
			for(; ix < thread_block_size && x + ix <= w - 16; ix += 16, src_ptr += 16, row_ptr += 16, dst_ptr += 16, buff_ptr += 16) {
				__m128 s1, s2, s3, s4;
				s1 = _mm_setzero_ps(); s2 = s1; s3= s1; s4 = s1;
				for(int k = 0; k < RADIUS; k++) {
					if(y < h - (k + 1)) {
						__m128 f = _mm_load_ss(by_ + k);
						f = _mm_shuffle_ps(f, f, 0);
						__m128i x0, x1, x2, x3, x4;
						x0 = _mm_loadu_si128((__m128i*)(src_ptr + (k + 1) * w));
						x1 = _mm_unpackhi_epi8(x0, z);
						x2 = _mm_unpacklo_epi8(x0, z);
						x3 = _mm_unpackhi_epi16(x2, z);
						x4 = _mm_unpacklo_epi16(x2, z);
						x2 = _mm_unpacklo_epi16(x1, z);
						x1 = _mm_unpackhi_epi16(x1, z);
						s1 = _mm_add_ps(s1, _mm_mul_ps(f, _mm_cvtepi32_ps(x1)));
						s2 = _mm_add_ps(s2, _mm_mul_ps(f, _mm_cvtepi32_ps(x2)));
						s3 = _mm_add_ps(s3, _mm_mul_ps(f, _mm_cvtepi32_ps(x3)));
						s4 = _mm_add_ps(s4, _mm_mul_ps(f, _mm_cvtepi32_ps(x4)));
						f = _mm_load_ss(ay + k);
						f = _mm_shuffle_ps(f, f, 0);
						__m128 y1, y2, y3, y4;
						y1 = _mm_loadu_ps(dst_ptr + (k + 1) * w + 12);
						y2 = _mm_loadu_ps(dst_ptr + (k + 1) * w + 8);
						y3 = _mm_loadu_ps(dst_ptr + (k + 1) * w + 4);
						y4 = _mm_loadu_ps(dst_ptr + (k + 1) * w);
						s1 = _mm_sub_ps(s1, _mm_mul_ps(f, y1));
						s2 = _mm_sub_ps(s2, _mm_mul_ps(f, y2));
						s3 = _mm_sub_ps(s3, _mm_mul_ps(f, y3));
						s4 = _mm_sub_ps(s4, _mm_mul_ps(f, y4));
					}
					else {
						__m128 f = _mm_set1_ps(by_[k] - ay[k]);
						__m128i x0, x1, x2, x3, x4;
						x0 = _mm_loadu_si128((__m128i*)(row_ptr));
						x1 = _mm_unpackhi_epi8(x0, z);
						x2 = _mm_unpacklo_epi8(x0, z);
						x3 = _mm_unpackhi_epi16(x2, z);
						x4 = _mm_unpacklo_epi16(x2, z);
						x2 = _mm_unpacklo_epi16(x1, z);
						x1 = _mm_unpackhi_epi16(x1, z);
						s1 = _mm_add_ps(s1, _mm_mul_ps(f, _mm_cvtepi32_ps(x1)));
						s2 = _mm_add_ps(s2, _mm_mul_ps(f, _mm_cvtepi32_ps(x2)));
						s3 = _mm_add_ps(s3, _mm_mul_ps(f, _mm_cvtepi32_ps(x3)));
						s4 = _mm_add_ps(s4, _mm_mul_ps(f, _mm_cvtepi32_ps(x4)));
						
					}	
				}
				_mm_storeu_ps(dst_ptr, s4);
				_mm_storeu_ps(dst_ptr + 4, s3);
				_mm_storeu_ps(dst_ptr + 8, s2);
				_mm_storeu_ps(dst_ptr + 12, s1);
				s1 = _mm_add_ps(s1, _mm_loadu_ps(buff_ptr + 12));
				s2 = _mm_add_ps(s2, _mm_loadu_ps(buff_ptr + 8));
				s3 = _mm_add_ps(s3, _mm_loadu_ps(buff_ptr + 4));
				s4 = _mm_add_ps(s4, _mm_loadu_ps(buff_ptr));
				_mm_storeu_ps(buff_ptr, s4);
				_mm_storeu_ps(buff_ptr + 4, s3);
				_mm_storeu_ps(buff_ptr + 8, s2);
				_mm_storeu_ps(buff_ptr + 12, s1);
			}
			for(; ix < thread_block_size && x + ix < w; ix++, src_ptr++, row_ptr++, dst_ptr++, buff_ptr++) {
				for(int k = 0; k < RADIUS; k++) {
					if(y < h - (k + 1)) {
						*dst_ptr += by_[k] * (*(src_ptr + (k + 1) * w)) - ay[k] * (*(dst_ptr + (k + 1) * w));
					} else {
						*dst_ptr += (by_[k] - ay[k]) * (*row_ptr);
					}
				}
				*buff_ptr += *dst_ptr;
			}
		}
	}
	
	#ifdef _OPENMP
	#pragma omp for schedule(static, h / NUM_THREADS)
	#endif
	for(int y = 0; y < h; y++) {
		int tid = omp_get_thread_num();
		float tmp[w + 2 * RADIUS * cn];
		float * src_ptr = tmp;
		float * buff_ptr = reinterpret_cast<float*>(buff.ptr<DST_TYPE>(y));
		float * row_ptr = reinterpret_cast<float*>(hrow.ptr<DST_TYPE>(tid)); 
		uchar * dst_ptr = reinterpret_cast<uchar*>(dst.ptr<SRC_TYPE>(y));
		memcpy(src_ptr + RADIUS * cn, buff_ptr, sizeof(float) * w);
		memcpy(row_ptr, src_ptr + RADIUS * cn, sizeof(float) * cn);
		memcpy(row_ptr + cn, row_ptr, sizeof(float) * cn);
		memcpy(row_ptr + 2 * cn, row_ptr, sizeof(float) * 2 * cn);
		memset(row_ptr + RADIUS * cn, 0, sizeof(float) * w);
		memcpy(src_ptr, row_ptr, sizeof(float) * RADIUS * cn);
		row_ptr += RADIUS * cn;
		src_ptr += RADIUS * cn;
		for(int x = 0; x < w; x++, src_ptr++, row_ptr++, dst_ptr++) {
		//	for(int k = 0; k < RADIUS; k++) {
		//		if(x >= k * cn) {
		//			*row_ptr += bx[k] * (*(src_ptr - k * cn));
		//		} else {
		//			*row_ptr += bx[k] * (*(row_ptr - k * cn));
		//		}
		//		*row_ptr -= ax[k] * (*(row_ptr - (k + 1) * cn));
		//	}
		//	*dst_ptr = saturate_cast<uchar>(*row_ptr);
			__m128 x0, h;
			x0 = _mm_set_ps(*(src_ptr - 3 * cn), *(src_ptr - 2 * cn), *(src_ptr - cn), *(src_ptr));
			x0 = _mm_mul_ps(x0, _mm_load_ps(bx));
			h = _mm_set_ps(*(row_ptr - 4 * cn), *(row_ptr - 3 * cn), *(row_ptr - 2 * cn), *(row_ptr - cn));
			h = _mm_mul_ps(h, _mm_load_ps(ax));
			x0 = _mm_sub_ps(x0, h);
			x0 = _mm_hadd_ps(x0, x0);
			x0 = _mm_hadd_ps(x0, x0);
			_mm_store_ss(row_ptr, x0);
			*dst_ptr = saturate_cast<uchar>(*row_ptr);
		}
		dst_ptr = reinterpret_cast<uchar*>(dst.ptr<SRC_TYPE>(y)) + w - 1;
		memcpy(row_ptr, src_ptr - cn, sizeof(float) * cn);
		memcpy(row_ptr + cn, row_ptr, sizeof(float) * cn);
		memcpy(row_ptr + 2 * cn, row_ptr, sizeof(float) * 2 * cn);
		memset(row_ptr - w, 0, sizeof(float) * w);
	
		memcpy(src_ptr, src_ptr - cn, sizeof(float) * cn);
		memcpy(src_ptr + cn, src_ptr, sizeof(float) * cn);
		memcpy(src_ptr + 2 * cn, src_ptr, sizeof(float) * 2 * cn);
//		memcpy(src_ptr, row_ptr, sizeof(float) * RADIUS * cn);
		src_ptr -= 1;
		row_ptr -= 1;
		for(int x = w - 1; x >= 0; x--, src_ptr--, row_ptr--, dst_ptr--) {
		//	for(int k = 0; k < RADIUS; k++) {
		//		if(x + (k + 1) * cn < w) {
		//			*row_ptr += bx_[k] * (*(src_ptr + (k + 1) * cn));
		//		} else {
		//			*row_ptr += bx_[k] * (*(row_ptr + (k + 1) * cn));
		//		}
		//		*row_ptr -= ax[k] * (*(row_ptr + (k + 1) * cn));
		//	}
		//	*dst_ptr = saturate_cast<uchar>(*row_ptr + *dst_ptr);
			__m128 x0, h;
			x0 = _mm_set_ps(*(src_ptr + 4 * cn), *(src_ptr + 3 * cn), *(src_ptr + 2 * cn), *(src_ptr + cn));
			x0 = _mm_mul_ps(x0, _mm_load_ps(bx_));
			h = _mm_set_ps(*(row_ptr + 4 * cn), *(row_ptr + 3 * cn), *(row_ptr + 2 * cn), *(row_ptr + cn));
			h = _mm_mul_ps(h, _mm_load_ps(ax));
			x0 = _mm_sub_ps(x0, h);
			x0 = _mm_hadd_ps(x0, x0);
			x0 = _mm_hadd_ps(x0, x0);
			_mm_store_ss(row_ptr, x0);
			*dst_ptr = saturate_cast<uchar>(*row_ptr + *dst_ptr);
		}
	}
	}
	t = omp_get_wtime() - t;
}

void fastGaussianIIR::apply(const Mat& src, Mat& dst) const
{
	if(src.channels() == 1) {
		apply_<uchar, float, 1>(src, dst);
	}
	else if(src.channels() == 2) {
		apply_<Vec2b, Vec2f, 2>(src, dst);
	}
	else if(src.channels() == 3) {
		apply_<Vec3b, Vec3f, 3>(src, dst);
	}
}

inline void fastGaussianIIR::getCoefficients_(double sigma, float _a[4], float _b[4], float _b_[4])
{
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
	complex<double> A[RADIUS][RADIUS];
	complex<double> rhs[RADIUS];
	int ipiv[RADIUS];
	for(int i = 0; i < RADIUS; i++) {
		A[i][0] = 1.0 / poles[i];
		rhs[i] = -1.0;
		if(i != 0) ai[i] *= 1.0 - poles[0] * A[i][0];
		for(int j = 1; j < RADIUS; j++) {
			if((j + 1) % 2 == 0) {
				A[i][j] = A[i][(j - 1) / 2] * A[i][(j - 1) / 2];
			} else {
				A[i][j] = A[i][(j - 1) / 2] * A[i][j / 2];
			}
			if(i != j) ai[i] *= 1.0 - poles[j] * A[i][0];
		}
	}
	int ret;
	if((ret = LAPACKE_zgetrf(LAPACK_ROW_MAJOR, RADIUS, RADIUS, reinterpret_cast<lapack_complex_double*>(A[0]), RADIUS, ipiv)) != 0) {
		fprintf(stderr, "ERROR: lapack routine LAPACKE_zgetrf() for solving a of simga = %f failed!\n", sigma);
		exit(-1);
	}
	if((ret = LAPACKE_zgetrs(LAPACK_ROW_MAJOR, 'N', RADIUS, 1, reinterpret_cast<lapack_complex_double*>(A[0]), RADIUS, ipiv, reinterpret_cast<lapack_complex_double*>(rhs), 1)) != 0) {
		fprintf(stderr, "ERROR: lapack routine LAPACKE_zgetrs() for solving a of sigma = %f failed!\n", sigma);
		exit(-1);
	}
	for(int i = 0; i < RADIUS; i++) {
		A[i][0] = 1.0;
		A[i][1] = 1.0 / poles[i];
		A[i][2] = A[i][1] * A[i][1];
		A[i][3] = A[i][2] * A[i][1];
	}
	if((ret = LAPACKE_zgetrf(LAPACK_ROW_MAJOR, RADIUS, RADIUS, reinterpret_cast<lapack_complex_double*>(A[0]), RADIUS, ipiv)) != 0) {
		fprintf(stderr, "ERROR: lapack routine LAPACKE_zgetrf() for solving b of simga = %f failed!\n", sigma);
		exit(-1);
	}
	if((ret = LAPACKE_zgetrs(LAPACK_ROW_MAJOR, 'N', RADIUS, 1, reinterpret_cast<lapack_complex_double*>(A[0]), RADIUS, ipiv, reinterpret_cast<lapack_complex_double*>(ai), 1)) != 0) {
		fprintf(stderr, "ERROR: lapack routine LAPACKE_zgetrs() for solving b of sigma = %f failed!\n", sigma);
		exit(-1);
	}
	for(int i = 0; i < RADIUS; i++) {
		_a[i] = real(rhs[i]);
		_b[i] = real(ai[i]) / (sqrt(2.0 * M_PI ) * sigma);
	}
	_b_[0] = _b[1] - _b[0] * _a[0];
	_b_[1] = _b[2] - _b[0] * _a[1];
	_b_[2] = _b[3] - _b[0] * _a[2];
	_b_[3] = - _b[0] * _a[3];
}

void fastGaussianIIR::getCoefficients()
{
	getCoefficients_(sigmaX, ax, bx, bx_);
	for(int i = 0; i < RADIUS; i++) {
#ifdef DEBUG
fprintf(stdout, "INFO: coefficients ax[%d] = %f, bx[%d] = %f, bx_[%d] = %f\n", i, ax[i], i, bx[i], i, bx_[i]);
#endif		
	}
	if(abs(sigmaX - sigmaY) < EPSILON) {
		memcpy(ay, ax, sizeof(float) * RADIUS);
		memcpy(by, bx, sizeof(float) * RADIUS);
		memcpy(by_, bx_, sizeof(float) * RADIUS);
	} else {
		getCoefficients_(sigmaY, ay, by, by_);
		for(int i = 0; i < RADIUS; i++) {
#ifdef DEBUG
fprintf(stdout, "INFO: coefficients ay[%d] = %f, by[%d] = %f, by_[%d] = %f\n", i, ay[i], i, by[i], i, by_[i]);
#endif		
		}
	}
}

void constTimeGaussianBlur(const Mat& src, Mat& dst, double sigmaX, double sigmaY, const fastGaussianIIR * gblur)
{
	double sigma = std::max(sigmaX, sigmaY);
	if(sigma <= 3 || gblur == NULL) {
		int kx = 2 * static_cast<int>(3.0 * sigmaX) + 1;
		int ky = 2 * static_cast<int>(3.0 * sigmaY) + 1;
		myGaussianBlur(src, dst, Size(kx, ky), sigmaX, sigmaY);
	} else {
		assert(std::abs(gblur->sigmaX - sigmaX) < EPSILON && std::abs(gblur->sigmaY - sigmaY) < EPSILON);
		gblur->apply(src, dst);
	}
}

void boxFilter(const Mat& src, Mat& dst, int radiusX, int radiusY)
{
	double ai = 1./ ((2.0 * radiusX + 1) * (2 * radiusY + 1));
	int h = src.rows; int w = src.cols;
	assert(h == dst.rows && w == dst.cols);
	int cn = src.channels();
	w *= cn;
	Mat_<double> sumArea(h, w, 0.0);
	double * sum_ptr = sumArea.ptr<double>(0);
	const uchar * src_ptr = src.ptr<uchar>(0);
	for(int i = 0; i < cn; i++, sum_ptr++, src_ptr++) {
		*sum_ptr = static_cast<double>(*src_ptr);
	}
	for(int x = cn; x < w; x++, sum_ptr++, src_ptr++) {
		*sum_ptr = *(sum_ptr - cn) + *src_ptr;
	}
	for(int y = 1; y < h; y++) {
		for(int x = 0; x < w; x++) { 
			if(x > cn) {
				*sum_ptr = *(sum_ptr - cn) + *(sum_ptr - w) - *(sum_ptr - w - cn) + *src_ptr;
			} else {
				*sum_ptr = *(sum_ptr - w) + *src_ptr;
			}		

			sum_ptr++;
			src_ptr++;	
		}
	}
	radiusX *= cn;
	for(int y = radiusY; y < h - radiusY; y++) {
		double * sum_ptr = sumArea.ptr<double>(y) + radiusX;
		uchar * dst_ptr = dst.ptr<uchar>(0) + y * w + radiusX;
		for(int x = radiusX; x < w - radiusX; x++, sum_ptr++, dst_ptr++) {
			double top(0.0), left(0.0), topLeft(0.0);
			if(y > radiusY) {
				top = *(sum_ptr + radiusX - (radiusY + 1) * w);
			}
			if(x >= radiusX + cn) {
				left = *(sum_ptr + radiusY * w - radiusX - cn);
			}
			if(x >= radiusX + cn && y > radiusY) {
				topLeft = *(sum_ptr - (radiusY + 1) * w - radiusX - cn);
			}
			*dst_ptr = saturate_cast<uchar>((*(sum_ptr + radiusY * w + radiusX) - top - left + topLeft) * ai);
		}
	}
	for(int y = 0; y < radiusY; y++) {
		const uchar * src_ptr = src.ptr<uchar>(0) + y * w;
		uchar * dst_ptr = dst.ptr<uchar>(0) + y * w;
		const uchar * src_bot_ptr = src_ptr + (h - radiusY) * w;
		uchar * dst_bot_ptr = dst_ptr + (h - radiusY) * w;
		for(int x = 0; x < w; x++, src_ptr++, dst_ptr++, src_bot_ptr++, dst_bot_ptr++) {
	//		*dst_ptr = saturate_cast<uchar>(0.5 * (*src_ptr) + 0.5 * (*(src_ptr + w)));
	//		*dst_bot_ptr = saturate_cast<uchar>(0.5 * (*src_bot_ptr) + 0.5 * (*(src_bot_ptr - w)));
			double val = (*src_ptr);
			double val_bot = (*src_bot_ptr);
			for(int k = 1; k <= radiusY; k++) {
				val += *(src_ptr + w * k);
				val_bot += *(src_bot_ptr - w * k);
			}
			*dst_ptr = saturate_cast<uchar>(val / (radiusY + 1.0));
			*dst_bot_ptr = saturate_cast<uchar>(val_bot / (radiusY + 1.0));
		}
	}
	for(int y = radiusY; y < h - radiusY; y++) {
		const uchar * src_ptr = src.ptr<uchar>(0) + y * w;
		uchar * dst_ptr = dst.ptr<uchar>(0) + y * w;
		const uchar * src_right_ptr = src_ptr + w - radiusX;
		uchar * dst_right_ptr = dst_ptr + w - radiusX;
		for(int x = 0; x < radiusX; x++, src_ptr++, dst_ptr++, src_right_ptr++, dst_right_ptr++) {
	//		*dst_ptr = saturate_cast<uchar>(0.5 * (*src_ptr) + 0.5 * (*(src_ptr + cn)));
	//		*dst_right_ptr = saturate_cast<uchar>(0.5 * (*src_right_ptr) + 0.5 * (*(src_right_ptr - cn)));
			double val = (*src_ptr);
			double val_right = (*src_right_ptr);
			for(int k = 1; k <= radiusX; k++) {
				val += *(src_ptr + k * cn);
				val_right += *(src_right_ptr - k * cn);
			}
			*dst_ptr = saturate_cast<uchar>(val / (radiusX + 1.0));
			*dst_right_ptr = saturate_cast<uchar>(val_right / (radiusX + 1.0));

		}
	}
}

/* bilateral filter */
#define DEPTH 255
class BilateralFilter {
public:
	BilateralFilter(double spSigmaX_, double spSimgaY_, double rangeSigma_, double factor_ = 25);
	~BilateralFilter() { };
	void apply(const Mat& src, Mat& dst);
public:
	double factor;
	double spSigmaX;
	double spSigmaY;
	double rangeSigma;
	double rangeKernelLookupTable[DEPTH];
};

BilateralFilter::BilateralFilter(double spSigmaX_, double spSigmaY_, double rangeSigma_, double factor_) : spSigmaX(spSigmaX_), spSigmaY(spSigmaY_), rangeSigma(rangeSigma_), factor(factor_) {
//	double twopisigmai = 1.0 / (sqrt(2.0 * M_PI) * rangeSigma);
	double twosigmasqi = 1.0 / (2.0 * rangeSigma * rangeSigma);
	for(int i = 0; i < DEPTH; i++) {
		rangeKernelLookupTable[i] = exp(- i * i * twosigmasqi);
	}
}

void BilateralFilter::apply(const Mat& src, Mat& dst)
{
	int nSamples = static_cast<int>(DEPTH / factor) + 1;
	vector<uchar> colorSamples(nSamples);
	Mat Wk(src.size(), src.type());
	Mat Jk(src.size(), src.type());
	Mat WkBlur(src.size(), src.type());
	Mat JkBlur(src.size(), src.type());
	vector<Mat> Jbk(nSamples);
	fastGaussianIIR gblur(spSigmaX, spSigmaY);
	int h = src.rows; int w= src.cols; int cn = src.channels();
	w *= cn;
//	#ifdef _OPENMP
//	#pragma omp parallel for num_threads(NUM_THREADS) schedule(static, nSamples / NUM_THREADS)
//	#endif
	double t0 = 0.0; double t1 = 0.0; double t2 = 0.0; double t3 = 0.0; double t4 = 0.0;
	for(int icolor = 0; icolor < nSamples; icolor++) {
		uchar color = saturate_cast<uchar>(icolor * factor);
		Jbk[icolor] = Mat(src.size(), src.type());
	//	Mat Wk(src.size(), src.type());
	//	Mat Jk(src.size(), src.type());
	//	Mat WkBlur(src.size(), src.type());
	//	Mat JkBlur(src.size(), src.type());

		t0 = omp_get_wtime();
		#ifdef _OPENMP
		#pragma omp parallel for num_threads(NUM_THREADS) schedule(static, h / NUM_THREADS)
		#endif
		for(int y = 0; y < h; y++) {
			const uchar * src_ptr = src.ptr<uchar>(0) + y * w;
			uchar * wk_ptr = Wk.ptr<uchar>(0) + y * w;
			uchar * jk_ptr = Jk.ptr<uchar>(0) + y * w;
			for(int x = 0; x < w; x++, src_ptr++, wk_ptr++, jk_ptr++) {
				uchar idx = saturate_cast<uchar>(std::abs(static_cast<int>(color) 
									- static_cast<int>(*src_ptr)));
				*wk_ptr = saturate_cast<uchar>(rangeKernelLookupTable[idx] * DEPTH);
				*jk_ptr = saturate_cast<uchar>(rangeKernelLookupTable[idx] * *src_ptr);
			}
		}
		t1 += omp_get_wtime() - t0;
		t0 = omp_get_wtime();
		constTimeGaussianBlur(Wk, WkBlur, spSigmaX, spSigmaY, &gblur);
		constTimeGaussianBlur(Jk, JkBlur, spSigmaX, spSigmaY, &gblur);
//		myBoxFilter(Wk, WkBlur, Size(2 * spSigmaX + 1, 2 * spSigmaY + 1));
//		myBoxFilter(Jk, JkBlur, Size(2 * spSigmaX + 1, 2 * spSigmaY + 1));
//		boxFilter(Wk, WkBlur, spSigmaX, spSigmaY);
//		boxFilter(Jk, JkBlur, spSigmaX, spSigmaY);
		t2 += omp_get_wtime() - t0;
		t0 = omp_get_wtime();
		#ifdef _OPENMP
		#pragma omp parallel for num_threads(NUM_THREADS) schedule(static, h / NUM_THREADS)
		#endif
		for(int y = 0; y < h; y++) {
			uchar * wk_ptr = WkBlur.ptr<uchar>(0) + y * w;
			uchar * jk_ptr = JkBlur.ptr<uchar>(0) + y * w;
			uchar * jbk_ptr = Jbk[icolor].ptr<uchar>(0) + y * w;
			for(int x = 0; x < w; x++, jbk_ptr++, wk_ptr++, jk_ptr++) {
				*jbk_ptr = saturate_cast<uchar>(DEPTH * static_cast<double>(*jk_ptr) / static_cast<double>(*wk_ptr));
			}
		}
		t3 += omp_get_wtime() - t0;
	}
	t0 = omp_get_wtime();
	#ifdef _OPENMP
	#pragma omp parallel for num_threads(NUM_THREADS) schedule(static, h / NUM_THREADS)
	#endif
	for(int y = 0; y < h; y++) {
		const uchar * src_ptr = src.ptr<uchar>(0) + y * w;
		uchar * dst_ptr = dst.ptr<uchar>(0) + y * w;
		for(int x = 0; x < w; x++, src_ptr++, dst_ptr++) {
			int i1 = static_cast<int>(*src_ptr / factor);
			int i2 = i1 < nSamples - 1 ? i1 + 1 : i1;
			uchar color1 = saturate_cast<uchar>(i1 * factor);
		//	uchar color2 = saturate_cast<uchar>(i2 * factor);
		//	double w1 = (1.0 * color2 - *src_ptr) / factor;	
			double w2 = (*src_ptr - 1.0 * color1) / factor;
			double w1 = 1.0 - w2;
			if(w1 < 0 || w2 < 0) cout << w1 << "\t" << w2 << "\n";
			uchar jbk1 = *(Jbk[i1].ptr<uchar>(0) + y * w + x);
			uchar jbk2 = *(Jbk[i2].ptr<uchar>(0) + y * w + x);
			*dst_ptr = saturate_cast<uchar>(jbk1 * w1 + jbk2 * w2);
		}
	}
	t4 = omp_get_wtime() - t0;
	cout << "t1 = " << t1 << "\n";
	cout << "t2 = " << t2 << "\n";
	cout << "t3 = " << t3 << "\n";
	cout << "t4 = " << t4 << endl;
//	dst = Jbk[128];

}

inline double weight(double x, double val = 150.0)
{
	double ret = x < 5 ? 1.0 : (0.5 + 0.5 * cos((x - 5.0) / (val - 5.0)));
	ret = x >= val ? 0.0 : ret;
	return ret; 
}

int main(int argc, char * argv[])
{
	if(argc < 6) {
		fprintf(stdout, "Usage: inputfile outputfile sigma rangesigma factor\n");
		return -1;
	}
	const char * inputimage = argv[1];
	const char * outputimage = argv[2];
	float sigma = atof(argv[3]);
	float rangesigma = atof(argv[4]);
	float factor = atof(argv[5]);
	Mat src = imread(inputimage);
	cout << src.size() << "\n";
//	resize(src, src, Size(3001, 2500));
	Mat dst(src.size(), src.type());
	fastGaussianIIR gblur(sigma, sigma);
	clock_t beg, end;
	double t;
	beg = clock();
	t = omp_get_wtime();
	constTimeGaussianBlur(src, dst, sigma, sigma, &gblur);
	gblur.apply(src, dst);
////	boxFilter(src, dst, 30, 30);
//	myBoxFilter(src, dst, Size(21, 21));
	t = omp_get_wtime() - t;
	end = clock();
	cout << "clock cycles of fastGaussianIIR(): " << static_cast<float>(end - beg) << "\n";
	cout << "elapsed time of fastGaussianIIR(): " << t << "s\n";
	imwrite(outputimage, dst);
//////	
//	Mat dst1(src.size(), src.type());
//	beg = clock();
//	t = omp_get_wtime();
//	GaussianBlur(src, dst1, Size(2 * (int)(3 * sigma) + 1, 2 * (int)(3 * sigma) + 1), sigma, sigma);
//	t = omp_get_wtime() - t;
//	end = clock();
//	cout << "clock cycles of opencv::GaussianBlur(): " << static_cast<float>(end - beg) << "\n";
//	cout << "elapsed time of opencv::GaussianBlur(): " << t << "s\n";
//	imwrite("gaussian_blur.jpg", dst1);	
//	imwrite("error.jpg", dst1 - dst);

	BilateralFilter blf(sigma, sigma, rangesigma, factor);
	Mat dst2(src.size(), src.type());
	blf.apply(src, dst2);
//      //bilateralFilter(src, dst2, 1.0, 32.0, 30.0, 1);
//	Mat back = imread("back.jpg");
//	for(int y = 0; y < src.rows; y++) {
//		Vec3b * src_ptr = dst2.ptr<Vec3b>(y);
//		Vec3b * bptr = back.ptr<Vec3b>(y);
//		for(int x = 0; x < src.cols; x++, src_ptr++, bptr++) {
//		//	int pred = (*bptr)[0] < 10 || (*bptr)[1] < 10 || (*bptr)[2] < 10;
//		//	(*src_ptr)[0] = pred ? (*src_ptr)[0] : saturate_cast<uchar>(0.5 * (*src_ptr)[0] + 0.5 * (*bptr)[0]);
//		//	(*src_ptr)[1] = pred ? (*src_ptr)[1] : saturate_cast<uchar>(0.5 * (*src_ptr)[1] + 0.5 * (*bptr)[1]);
//		//	(*src_ptr)[2] = pred ? (*src_ptr)[2] : saturate_cast<uchar>(0.5 * (*src_ptr)[2] + 0.5 * (*bptr)[2]);
//			(*src_ptr)[0] = saturate_cast<uchar>((*src_ptr)[0] * weight((*bptr)[0]) + (*bptr)[0] * (1.0 - weight((*bptr)[0])));
//			(*src_ptr)[1] = saturate_cast<uchar>((*src_ptr)[1] * weight((*bptr)[1]) + (*bptr)[1] * (1.0 - weight((*bptr)[1])));
//			(*src_ptr)[2] = saturate_cast<uchar>((*src_ptr)[2] * weight((*bptr)[2]) + (*bptr)[2] * (1.0 - weight((*bptr)[2])));
//	//		*src_ptr = *src_ptr + (*bptr);
//		}
//	}
////	dst2 = dst2 + back;
//
////	Mat dst3(src.size(), src.type());
////	myBoxFilter(dst2, dst3, Size(3, 3));
////	myBoxFilter(dst2, dst2, Size(3, 3));
	imwrite("blf.jpg", dst2);
//	__m128 a1 = {1.f, 1.f, 1.f, 1.f};
//	__m128 a2 = {2.f, 2.f, 2.f, 2.f};
//	__m128 a3 = {3.f, 3.f, 3.f, 3.f};
//	__m128 a4 = {4.f, 4.f, 4.f, 4.f};
//	__m128 b = _mm_blend_ps(a1, a2, 10);
//	__m128 c = _mm_blend_ps(a3, a4, 10);
//	c = _mm_blend_ps(b, c, 12);
//	float tmp[4] __attribute__((aligned(16)));
//	_mm_store_ps(tmp, c);
//	cout << tmp[0] << ", " << tmp[1] << ", " << tmp[2] << ", " << tmp[3] << "\n";

	return 0;
}
