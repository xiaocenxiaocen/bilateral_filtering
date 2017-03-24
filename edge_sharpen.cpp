/* @file: edge_sharpen.cpp
 * @brief: 
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

#include <mmintrin.h> 	// MMX
#include <xmmintrin.h>	// SSE
#include <smmintrin.h>
#include <nmmintrin.h>
#include <emmintrin.h>	// SSE2
#include <immintrin.h>	// AVX
//#include <intrin.h>

using namespace cv; // all the new API is put into "cv" namespace. Export its content
using namespace std;

class EdgeSharpen {
public:
	virtual void Sharpen(const Mat& src, Mat& dst) const = 0;
};

class LaplaceEdgeSharpen : public EdgeSharpen {
public:
	virtual void Sharpen(const Mat& src, Mat& dst) const;
};

void LaplaceEdgeSharpen::Sharpen(const Mat& src, Mat& dst) const
{
	int h = src.rows;
	int w = src.cols;
	int cn = src.channels();
	assert(src.type() == dst.type() && src.size() == dst.size());

	int threadNum = omp_get_max_threads();
	w *= cn;
	__m128 cent = _mm_set1_ps(5.0);
	__m128i z = _mm_setzero_si128();
#ifdef DEBUG
fprintf(stdout, "DEBUG: num threads = %d, file = %s, line = %d\n", threadNum, __FILE__, __LINE__);
#endif
	#ifdef _OPENMP
	#pragma omp parallel for num_threads(threadNum) schedule(static, (h - 1) / threadNum) shared(src, dst)
	#endif
	for(int y = 1; y < h - 1; y++) {
		const uchar * src_ptr = src.ptr<uchar>(0) + y * w + cn;
		uchar * dst_ptr = dst.ptr<uchar>(0) + y * w + cn;
		int x = cn;
		for(; x <= w - cn - 16; x += 16, src_ptr += 16, dst_ptr += 16) {
			__m128i x0, x1, x2, x3, x4, y0;
			x0 = _mm_loadu_si128((__m128i*)(src_ptr));
			x1 = _mm_unpackhi_epi8(x0, z);
			x2 = _mm_unpacklo_epi8(x0, z);
			x3 = _mm_unpackhi_epi16(x2, z);
			x4 = _mm_unpacklo_epi16(x2, z);
			x2 = _mm_unpacklo_epi16(x1, z);
			x1 = _mm_unpackhi_epi16(x1, z);
			__m128 s1, s2, s3, s4;
			s1 = _mm_mul_ps(cent, _mm_cvtepi32_ps(x1));
			s2 = _mm_mul_ps(cent, _mm_cvtepi32_ps(x2));
			s3 = _mm_mul_ps(cent, _mm_cvtepi32_ps(x3));
			s4 = _mm_mul_ps(cent, _mm_cvtepi32_ps(x4));			
			
			x0 = _mm_loadu_si128((__m128i*)(src_ptr + cn));
			y0 = _mm_loadu_si128((__m128i*)(src_ptr - cn));
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
			s1 = _mm_sub_ps(s1, _mm_cvtepi32_ps(x1));
			s2 = _mm_sub_ps(s2, _mm_cvtepi32_ps(x2));
			s3 = _mm_sub_ps(s3, _mm_cvtepi32_ps(x3));
			s4 = _mm_sub_ps(s4, _mm_cvtepi32_ps(x4));
		
			x0 = _mm_loadu_si128((__m128i*)(src_ptr + w));
			y0 = _mm_loadu_si128((__m128i*)(src_ptr - w));
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
			s1 = _mm_sub_ps(s1, _mm_cvtepi32_ps(x1));
			s2 = _mm_sub_ps(s2, _mm_cvtepi32_ps(x2));
			s3 = _mm_sub_ps(s3, _mm_cvtepi32_ps(x3));
			s4 = _mm_sub_ps(s4, _mm_cvtepi32_ps(x4));
			
			x1 = _mm_cvttps_epi32(s4);
			x2 = _mm_cvttps_epi32(s3);
			x3 = _mm_cvttps_epi32(s2);
			x4 = _mm_cvttps_epi32(s1);
			x1 = _mm_packs_epi32(x1, x2);
			x2 = _mm_packs_epi32(x3, x4);
			x1 = _mm_packus_epi16(x1, x2);
			_mm_storeu_si128((__m128i*)dst_ptr, x1);	
		}
		for(; x < w - cn; x++, src_ptr++, dst_ptr++) {
			float sharp = 5.0 * (*src_ptr) - (*(src_ptr + cn) + *(src_ptr - cn) + *(src_ptr + w) + *(src_ptr - w));
			*dst_ptr = saturate_cast<uchar>(sharp);
		}
	}
}

int main(int argc, char* argv[])
{
	if(argc < 3) {
		fprintf(stdout, "Usage: inputfile outputfile\n");
		return -1;
	}
	const char * inputimage = argv[1];
	const char * outputimage = argv[2];
	Mat src = imread(inputimage);
	Mat dst(src.size(), src.type());
	LaplaceEdgeSharpen sharpen;
	sharpen.Sharpen(src, dst);
	imwrite(outputimage, dst);
	return 0;
}
