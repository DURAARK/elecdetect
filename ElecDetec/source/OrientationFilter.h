/*
 * OrientationFilter.h
 *
 *  Created on: Oct 6, 2014
 *      Author: test
 */

#ifndef ORIENTATIONFILTER_H_
#define ORIENTATIONFILTER_H_

#include <opencv2/opencv.hpp>

#include "Defines.h"
#include "Utils.h"
#include "VisionModule.h"
#include "VisionData.h"

using namespace std;
using namespace cv;

class COrientationFilter: public CVisionModule
{
private:
	int orientation_bin_idx_;
	int n_orientation_bins_;

	float x_weight_; // resulting weight for x-gradient-component
	float y_weight_; // resulting weight for y-gradient-component (x^2+y^2=1)

	float to_zero_threshold_;

	// Sobel parameters
	int scale_;
	int delta_;
	int ddepth_;

	COrientationFilter();
public:
	COrientationFilter(MODULE_CONSTRUCTOR_SIGNATURE);
	virtual ~COrientationFilter();

	virtual CVisionData* exec();
	virtual void save(FileStorage& fs) const;
	virtual void load(FileStorage& fs);
};

#endif /* ORIENTATIONFILTER_H_ */
