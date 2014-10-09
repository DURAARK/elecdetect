/*
 * GradientOrientationFeatures.h
 *
 *  Created on: Oct 9, 2014
 *      Author: test
 */

#ifndef GRADIENTORIENTATIONFEATURES_H_
#define GRADIENTORIENTATIONFEATURES_H_

#include "Utils.h"
#include "Defines.h"
#include "VisionModule.h"

class CGradientOrientationFeatures: public CVisionModule
{
	typedef pair<Rect, Rect> RectPair;

private:
	CGradientOrientationFeatures();

	int n_tests_;
	int symm_percentage_;

	// Threshold to reduce noise
	float mag_threshold_;

	// Sobel parameter
	int ddepth_ = CV_32F;

	vector<RectPair> wavelet_pairs_;

	void initWavelets();


public:
	CGradientOrientationFeatures(MODULE_CONSTRUCTOR_SIGNATURE);
	virtual ~CGradientOrientationFeatures();

	virtual CVisionData* exec();
	virtual void save(FileStorage& fs) const;
	virtual void load(FileStorage& fs);
};

#endif /* GRADIENTORIENTATIONFEATURES_H_ */
