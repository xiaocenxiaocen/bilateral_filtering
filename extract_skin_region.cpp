/* @file: extract_skin_region.cpp
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
#include <vector>

#include <mmintrin.h> 	// MMX
#include <xmmintrin.h>	// SSE
#include <smmintrin.h>
#include <nmmintrin.h>
#include <emmintrin.h>	// SSE2
#include <immintrin.h>	// AVX
//#include <intrin.h>

using namespace cv; // all the new API is put into "cv" namespace. Export its content
using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979
#endif
#define RAD2DEG (180.0 / M_PI)

class SkinRegionExtractor {
public:
	virtual void Extractor(const Mat& src, Mat& dst, Mat& back) const = 0;
};

class LogOpponentExtractor : public SkinRegionExtractor {
public:
	virtual void Extractor(const Mat& src, Mat& dst, Mat& back) const;
	static inline bool LogOpponentPredictor(uchar r, uchar g, uchar b);
};

inline bool LogOpponentExtractor::LogOpponentPredictor(uchar r, uchar g, uchar b)
{
	double I = log(g);
	double Rg = log(r) - I;
	double By = log(b) - (I + 0.5 * Rg);
	double hue = atan2(Rg, By) * RAD2DEG;
	I = 0.5957 * r - 0.2745 * g - 0.3213 * b;
	bool pred = I >= 20 && I <= 90 && (hue >= 100 && hue <= 150);
	return pred;
}

class HsvExtractor : public SkinRegionExtractor {
public:
	virtual void Extractor(const Mat& src, Mat& dst, Mat& back) const;
	static inline bool HsvPredictor(uchar r, uchar g, uchar b);
};

class RGSpaceExtractor : public SkinRegionExtractor {
public:
	virtual void Extractor(const Mat& src, Mat& dst, Mat& back) const;
	static inline bool RGSpacePredictor(uchar r, uchar g, uchar b);
};

class RGBLimitExtractor : public SkinRegionExtractor {
public:
	virtual void Extractor(const Mat& src, Mat& dst, Mat& back) const;
	static inline bool RGBLimitPredictor(uchar r, uchar g, uchar b);
};

#define EPSILON 1e-14
#define INVALID_S 999

void LogOpponentExtractor::Extractor(const Mat& src, Mat& dst, Mat& back) const
{
	int h = src.rows;
	int w = src.cols;
	int cn = src.channels();
	assert(src.type() == dst.type() && src.size() == dst.size() && src.type() == back.type() && src.size() == back.size() && cn == 3);

	int threadNum = omp_get_max_threads();
	fprintf(stdout, "INFO: num threads = %d, file = %s, line = %d\n", threadNum, __FILE__, __LINE__);
	#ifdef _OPENMP
	#pragma omp parallel for num_threads(threadNum) schedule(static, h / threadNum) shared(src, dst, back)
	#endif	
	for(int y = 0; y < h; y++) {
		const Vec3b * src_ptr = src.ptr<Vec3b>(y);
		Vec3b * dst_ptr = dst.ptr<Vec3b>(y);
		Vec3b * bptr = back.ptr<Vec3b>(y);
		for(int x = 0; x < w; x++, src_ptr++, dst_ptr++, bptr++) {
			bool pred1 = LogOpponentExtractor::LogOpponentPredictor((*src_ptr)[2], (*src_ptr)[1], (*src_ptr)[0]);
			bool pred2 = HsvExtractor::HsvPredictor((*src_ptr)[2], (*src_ptr)[1], (*src_ptr)[0]);
		//	bool pred3 = RGSpaceExtractor::RGSpacePredictor((*src_ptr)[2], (*src_ptr)[1], (*src_ptr)[0]);
		//	bool pred4 = RGBLimitExtractor::RGBLimitPredictor((*src_ptr)[2], (*src_ptr)[1], (*src_ptr)[0]);
			bool pred = pred1;
		//	bool pred = (0.3 * pred1 + 0.5 * pred2 + 0.2 * pred3 + 0.2 * pred4) > 0.5;
			*dst_ptr = pred ? *src_ptr : Vec3b(0x0, 0x0, 0x0);
			*bptr = pred ? Vec3b(0x0, 0x0, 0x0) : *src_ptr;
		}
	}
}

inline bool HsvExtractor::HsvPredictor(uchar r, uchar g, uchar b)
{
	double I = 0.5957 * r - 0.2745 * g - 0.3213 * b;
	double V = std::max(b, std::max(g, r));
	double D = V - std::min(b, std::min(g, r));
	D = D > EPSILON ? D : 1.0;
	double S = V > EPSILON ? D / V : INVALID_S;
	double H = 0.0;
	uchar uiV = saturate_cast<uchar>(V);
	bool rpred = r == uiV;
	bool gpred = (!rpred) && g == uiV;
	bool bpred = (!rpred) && (!gpred) && b == uiV;		
	H = rpred ? (g - b) / (6.0 * D) : H;
	H = gpred ? (2.0 - r + b) / (6.0 * D) : H;
	H = bpred ? (4.0 - g + b) / (6.0 * D) : H;
	bool pred = (S >= 0.20 && S <= 0.75) && V > 0.35 && H >= 0 && H <=50 && I <= 90 && I >= 20;
	return pred;
}

void HsvExtractor::Extractor(const Mat& src, Mat& dst, Mat& back) const
{
	int h = src.rows;
	int w = src.cols;
	int cn = src.channels();
	assert(src.type() == dst.type() && src.size() == dst.size() && src.type() == back.type() && src.size() == back.size() && cn == 3);

	int threadNum = omp_get_max_threads();
	fprintf(stdout, "INFO: num threads = %d, file = %s, line = %d\n", threadNum, __FILE__, __LINE__);
	#ifdef _OPENMP
	#pragma omp parallel for num_threads(threadNum) schedule(static, h / threadNum) shared(src, dst, back)
	#endif
	for(int y = 0; y < h; y++) {
		const Vec3b * src_ptr = src.ptr<Vec3b>(y);
		Vec3b * dst_ptr = dst.ptr<Vec3b>(y);
		Vec3b * bptr = back.ptr<Vec3b>(y);
		for(int x = 0; x < w; x++, src_ptr++, dst_ptr++, bptr++) {
//		//	int pred = (S >= 0.20 && S <= 0.68) && V > 0.35 && H >= 0 && H <=50;
			bool pred = HsvExtractor::HsvPredictor((*src_ptr)[2], (*src_ptr)[1], (*src_ptr)[0]);
			*dst_ptr = pred ? *src_ptr : Vec3b(0x0, 0x0, 0x0);
			*bptr = pred ? Vec3b(0x0, 0x0, 0x0) : *src_ptr;
		}
	}
}


inline bool RGSpaceExtractor::RGSpacePredictor(uchar r, uchar g, uchar b)
{
	static const double cup[] = {-1.8423, 1.5924, 0.0422};
	static const double cdown[] = {-0.7279, 0.6066, 0.1766};
	double s = r + g + b;
	double dr = r / s;
	double dg = g / s;
	double gup = cup[0] * dr * dr + cup[1] * dr + cup[2];
	double gdown = cdown[0] * dr * dr + cdown[1] * dr + cdown[2];
	double wr = (dr - 0.33) * (dr - 0.33) + (dg - 0.33) * (dg - 0.33);
	bool pred = dg < gup && dg > gdown && wr > 0.004;
	return pred;
}

void RGSpaceExtractor::Extractor(const Mat& src, Mat& dst, Mat& back) const
{
	int h = src.rows;
	int w = src.cols;
	int cn = src.channels();
	assert(src.type() == dst.type() && src.size() == dst.size() && src.type() == back.type() && src.size() == back.size() && cn == 3);

	int threadNum = omp_get_max_threads();
	fprintf(stdout, "INFO: num threads = %d, file = %s, line = %d\n", threadNum, __FILE__, __LINE__);
	#ifdef _OPENMP
	#pragma omp parallel for num_threads(threadNum) schedule(static, h / threadNum) shared(src, dst, back)
	#endif
	for(int y = 0; y < h; y++) {
		const Vec3b * src_ptr = src.ptr<Vec3b>(y);
		Vec3b * dst_ptr = dst.ptr<Vec3b>(y);
		Vec3b * bptr = back.ptr<Vec3b>(y);
		for(int x = 0; x < w; x++, src_ptr++, dst_ptr++, bptr++) {
//		//	int pred = (S >= 0.20 && S <= 0.68) && V > 0.35 && H >= 0 && H <=50;
			bool pred = RGSpaceExtractor::RGSpacePredictor((*src_ptr)[2], (*src_ptr)[1], (*src_ptr)[0]);
			*dst_ptr = pred ? *src_ptr : Vec3b(0x0, 0x0, 0x0);
			*bptr = pred ? Vec3b(0x0, 0x0, 0x0) : *src_ptr;
		}
	}
}

inline bool RGBLimitExtractor::RGBLimitPredictor(uchar r, uchar g, uchar b)
{
	bool pred = (r > 95 && g > 40 && b > 20 && (r - b) > 15 && (r - g) > 15)
		 || (r > 200 && g > 210 && b > 170 && std::abs(r - b) <= 15 && r > b && g > b);
	return pred;
}

void RGBLimitExtractor::Extractor(const Mat& src, Mat& dst, Mat& back) const
{
	int h = src.rows;
	int w = src.cols;
	int cn = src.channels();
	assert(src.type() == dst.type() && src.size() == dst.size() && src.type() == back.type() && src.size() == back.size() && cn == 3);

	int threadNum = omp_get_max_threads();
	fprintf(stdout, "INFO: num threads = %d, file = %s, line = %d\n", threadNum, __FILE__, __LINE__);
	#ifdef _OPENMP
	#pragma omp parallel for num_threads(threadNum) schedule(static, h / threadNum) shared(src, dst, back)
	#endif
	for(int y = 0; y < h; y++) {
		const Vec3b * src_ptr = src.ptr<Vec3b>(y);
		Vec3b * dst_ptr = dst.ptr<Vec3b>(y);
		Vec3b * bptr = back.ptr<Vec3b>(y);
		for(int x = 0; x < w; x++, src_ptr++, dst_ptr++, bptr++) {
//		//	int pred = (S >= 0.20 && S <= 0.68) && V > 0.35 && H >= 0 && H <=50;
			bool pred = RGBLimitExtractor::RGBLimitPredictor((*src_ptr)[2], (*src_ptr)[1], (*src_ptr)[0]);
			*dst_ptr = pred ? *src_ptr : Vec3b(0x0, 0x0, 0x0);
			*bptr = pred ? Vec3b(0x0, 0x0, 0x0) : *src_ptr;
		}
	}
}

void Diffusion(const Mat& src, const Mat& seed, const Mat& edge, Mat& dst)
{
	int h = src.rows;
	int w = src.cols;
	int cn = src.channels();
	assert(src.type() == dst.type() && src.size() == dst.size() && cn == 3);

	int threadNum = omp_get_max_threads();
	fprintf(stdout, "INFO: num threads = %d, file = %s, line = %d\n", threadNum, __FILE__, __LINE__);
	#ifdef _OPENMP
	#pragma omp parallel for num_threads(threadNum) schedule(static, h / threadNum) shared(src, dst)
	#endif
	for(int y = 0; y < h; y++) {
		const Vec3b * src_ptr = src.ptr<Vec3b>(y);
		const Vec3b * seed_ptr = seed.ptr<Vec3b>(y);
		Vec3b * dst_ptr = dst.ptr<Vec3b>(y);
		for(int x = 0; x < w; x++, src_ptr++, dst_ptr++, seed_ptr++) {
			if((*seed_ptr)[0] != 0x0 && (*seed_ptr)[1] != 0x0 && (*seed_ptr)[2] != 0x0) {
				*dst_ptr = *src_ptr;
				int step = 1;
				while(x + step < w && edge.at<uchar>(y, x + step) != 0xff) {
					dst_ptr[step] = src_ptr[step];	
					step++;
				}
				step = 1;
				while(x + step < w && y + step < h && edge.at<uchar>(y + step, x + step) != 0xff) {
					dst_ptr[step * w + step] = src_ptr[step * w + step];	
					step++;
				}
				step = 1;
				while(y + step < h && edge.at<uchar>(y + step, x) != 0xff) {
					dst_ptr[step * w] = src_ptr[step * w];	
					step++;
				}
				step = 1;
				while(x - step >= 0 && y + step < h && edge.at<uchar>(y + step, x - step) != 0xff) {
					dst_ptr[step * w - step] = src_ptr[step * w - step];	
					step++;
				}
				step = 1;
				while(x - step >= 0 && edge.at<uchar>(y, x - step) != 0xff) {
					dst_ptr[-step] = src_ptr[- step];	
					step++;
				}
				step = 1;
				while(x - step >= 0 && y - step >= 0 && edge.at<uchar>(y - step, x - step) != 0xff) {
					dst_ptr[-step * w - step] = src_ptr[- step * w - step];	
					step++;
				}
				step = 1;
				while(y - step >=0 && edge.at<uchar>(y - step, x) != 0xff) {
					dst_ptr[- step * w] = src_ptr[ - step * w];	
					step++;
				}
				step = 1;
				while(x + step < w && y - step >= 0 && edge.at<uchar>(y - step, x + step) != 0xff) {
					dst_ptr[- step * w + step] = src_ptr[ - step * w + step];	
					step++;
				}
			}
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
	Mat back(src.size(), src.type());
	LogOpponentExtractor xtrct;
//	HsvExtractor xtrct;
//	RGSpaceExtractor xtrct;
//	RGBLimitExtractor xtrct;
	xtrct.Extractor(src, dst, back);
//	Mat edge = imread("edge.jpg");
//	Diffusion(src, dst1, edge, dst2);
//	imwrite("skin3.jpg", dst2);
	imwrite(outputimage, dst);
	imwrite("back.jpg", back);
	return 0;
}
