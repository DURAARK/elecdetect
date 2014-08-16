/*
 * DistanceTransform.cpp
 *
 *  Created on: Jun 30, 2014
 *      Author: test
 */

#include "DistanceTransform.h"

CDistanceTransform::CDistanceTransform(int inchain_input_signature)
{
	module_print_name_ = "Distance Transform";
	required_input_signature_mask_ = DATA_TYPE_IMAGE | CV_8UC1; // takes single channel (binary) image as input only
	output_type_ = DATA_TYPE_IMAGE | CV_32FC1;

	if(inchain_input_signature != required_input_signature_mask_)
	{
		data_converter_ = new CDataConverter(inchain_input_signature, required_input_signature_mask_);
	}
}

CDistanceTransform::~CDistanceTransform()
{

}

void CDistanceTransform::exec(const CVisionData& input_data, CVisionData& output_data)
{
	CVisionData working_data(input_data.data(), input_data.getType());
	if(data_converter_)
	{
		data_converter_->convert(working_data);
	}

	assert(working_data.getSignature == required_input_signature_mask_);

	Mat working_img = working_data.data();

	// check if image is binary, otherwise apply threshold
	bool is_binary = countNonZero(working_img == 255) + countNonZero(working_img == 0) == working_img.cols * working_img.rows;
	if(!is_binary)
	{
		if(working_img.channels() != 1)
			cvtColor(working_img, working_img, CV_BGR2GRAY);
//		threshold(temp, temp, 100, 255, CV_THRESH_BINARY);
		adaptiveThreshold(working_img, working_img, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 101, -50);
	}

	distanceTransform(255-working_img, working_img, CV_DIST_L2, 3);
	output_data.assignData(working_img, DATA_TYPE_IMAGE);
	//output_data.mat_.convertTo(output_data.mat_, CV_8UC1, 2, 0);
}

void CDistanceTransform::save(FileStorage& fs) const
{

}
void CDistanceTransform::load(FileStorage& fs)
{

}
