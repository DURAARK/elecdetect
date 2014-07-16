/*
 * DistanceTransform.cpp
 *
 *  Created on: Jun 30, 2014
 *      Author: test
 */

#include "DistanceTransform.h"

CDistanceTransform::CDistanceTransform()
{
	module_name_ = "DistTr";

}

CDistanceTransform::~CDistanceTransform()
{

}

void CDistanceTransform::exec(std::vector<CVisionData*>& data) throw(VisionDataTypeException)
{
	CMat* img0_ptr = (CMat*)data.back();
	CMat* out_img = new CMat();

	Mat temp = img0_ptr->mat_;

	// check if iamge is binary, otherwise apply threshold
	bool is_binary = countNonZero(temp ==255) + countNonZero(temp == 0) == temp.cols * temp.rows;
	if(!is_binary)
	{
		if(temp.channels() != 1)
			cvtColor(temp, temp, CV_BGR2GRAY);
//		threshold(temp, temp, 100, 255, CV_THRESH_BINARY);
		adaptiveThreshold(temp, temp, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 101, -50);
	}

	distanceTransform(255-temp, out_img->mat_, CV_DIST_L2, 3);
	out_img->mat_.convertTo(out_img->mat_, CV_8UC1, 2, 0);

	imshow("test", out_img->mat_);
	waitKey(0);

	data.push_back(out_img);
}

void CDistanceTransform::save(FileStorage& fs) const
{

}
void CDistanceTransform::load(FileStorage& fs)
{

}
