/*
 * ElecDetec: OrientationFilter.h
 *
 *  Created on: Feb, 2015
 *      Author: Robert Viehauser
 */

#ifndef ORIENTATIONFILTER_H_
#define ORIENTATIONFILTER_H_

#include <opencv2/opencv.hpp>
#include <Defines.h>
#include <Utils.h>

using namespace std;
using namespace cv;

class COrientationFilter
{
public:
    enum Direction { HORIZ, VERT };

private:
    Direction direction_;

	float to_zero_threshold_;

	// Sobel parameters
	int scale_;
	int delta_;
	int ddepth_;

	COrientationFilter();
public:
    COrientationFilter(const Direction& direction);
    ~COrientationFilter();

    void filterImage(const Mat& input_img, Mat& filtered_img);
};

#endif /* ORIENTATIONFILTER_H_ */
