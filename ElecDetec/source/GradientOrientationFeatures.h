/*
 * ElecDetec: GradientOrientationFeatures.h
 *
 *  Created on: Feb, 2015
 *      Author: Robert Viehauser
 */

#ifndef GRADIENTORIENTATIONFEATURES_H_
#define GRADIENTORIENTATIONFEATURES_H_

#include <Defines.h>
#include <Utils.h>


// algorithm parameters:
// n_tests: number of rect tests (int)
// symm_ratio: ratio of symmetric testpairs (float)
// pair_ratio: ratio of pair-wise tests (in opposite to single rects) (float)
// min_rect_sz: minimum size (border length) of a haar-like feature (int)
// max_rect_sz: maximum size (border length) of a haar-like feature (int)
// mag_threshold: gradient magnitude threshold to supress noise (float)

#define GOF_N_TESTS                                     1000
#define GOF_SYMM_RATIO                                   0.2
#define GOF_PAIR_RATIO                                   1.0
#define GOF_MIN_RECT_SZ                                    4
#define GOF_MAG_THRESHOLD                               10.0

using namespace std;
using namespace cv;

extern int _PATCH_WINDOW_SIZE_;

class CGradientOrientationFeatures
{
	typedef pair<Rect, Rect> RectPair;

private:
	int n_tests_;
    float symm_ratio_;
    float pair_ratio_;
    int min_rect_sz_;
    int max_rect_sz_;
	float mag_threshold_;

	vector<RectPair> wavelet_pairs_;

	void initWavelets();

public:
    CGradientOrientationFeatures();
    ~CGradientOrientationFeatures();

    void extractFeatureVector(const Mat& input_img, vector<float>& output_vec);
    void save(FileStorage& fs) const;
    void load(FileNode& _node);
};

#endif /* GRADIENTORIENTATIONFEATURES_H_ */
