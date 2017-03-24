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

class SkinWhiten {
public:
	virtual void Whiten(const Mat& src, Mat& dst) const = 0;
};

#define DEPTH 256
class LogCurveWhiten : public SkinWhiten {
public:
	LogCurveWhiten(double beta_ = 5.0) : beta(beta_) {
		for(int color = 0; color < DEPTH; color++) {
			double w = color / (DEPTH - 1.0);
			double v = log(w * (beta - 1.0) + 1.0) / log(beta);
			colorMap[color] = saturate_cast<uchar>(v * 255);
		}
	}
	virtual void Whiten(const Mat& src, Mat& dst) const;
public:
	double beta;
	uchar colorMap[DEPTH];
};

void LogCurveWhiten::Whiten(const Mat& src, Mat& dst) const
{
	int h = src.rows;
	int w = src.cols;
	int cn = src.channels();
	assert(src.type() == dst.type() && src.size() == dst.size());

	int threadNum = omp_get_max_threads();
	w *= cn;
#ifdef DEBUG
fprintf(stdout, "DEBUG: num threads = %d, file = %s, line = %d\n", threadNum, __FILE__, __LINE__);
#endif
	#ifdef _OPENMP
	#pragma omp parallel for num_threads(threadNum) schedule(static, h / threadNum) shared(src, dst)
	#endif
	for(int y = 0; y < h; y++) {
		const uchar * src_ptr = src.ptr<uchar>(0) + y * w;
		uchar * dst_ptr = dst.ptr<uchar>(0) + y * w;

		for(int x = 0; x < w; x++, src_ptr++, dst_ptr++) {
			*dst_ptr = colorMap[*src_ptr];
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
	LogCurveWhiten whiten(6.0);
	whiten.Whiten(src, dst);
	imwrite(outputimage, dst);
	return 0;
}
