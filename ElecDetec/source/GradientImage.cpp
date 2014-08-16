/*
 * GradientImage.cpp
 *
 *  Created on: Jun 30, 2014
 *      Author: test
 */

#include "GradientImage.h"

CGradientImage::CGradientImage(int inchain_input_signature)
{
	module_print_name_ = "Gradient";
	required_input_signature_mask_ = DATA_TYPE_IMAGE | CV_8UC1; // takes single channel grayscale image as input
	output_type_ = DATA_TYPE_IMAGE | CV_8UC1;

	if(inchain_input_signature != required_input_signature_mask_)
	{
		data_converter_ = new CDataConverter(inchain_input_signature, required_input_signature_mask_);
	}

	scale_ = 3;
	delta_ = 0;
	ddepth_ = CV_16S;
}

CGradientImage::~CGradientImage()
{

}

void CGradientImage::exec(const CVisionData& input_data, CVisionData& output_data)
{
	CVisionData working_data(input_data.data(), input_data.getType());
	if(data_converter_)
	{
		data_converter_->convert(working_data);
	}
	Mat out_img;

	Mat img0_gray, grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	// Blur and Grayscale
	GaussianBlur(working_data.data(), img0_gray, Size(3,3), 0, 0, BORDER_DEFAULT);
	cvtColor(img0_gray, img0_gray, CV_RGB2GRAY);

	// Gradient X
	Sobel(img0_gray, grad_x, ddepth_, 1, 0, 3, scale_, delta_, BORDER_DEFAULT);
	// Gradient Y
	Sobel(img0_gray, grad_y, ddepth_, 0, 1, 3, scale_, delta_, BORDER_DEFAULT);

	// Conversion to absolute gradient image
	convertScaleAbs(grad_x, abs_grad_x );
	convertScaleAbs(grad_y, abs_grad_y );

	// superposition of both directions
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, out_img);

	//normalize(out_img->mat_, out_img->mat_, 0, 255, NORM_MINMAX);
	output_data.assignData(out_img, DATA_TYPE_IMAGE);
}

void CGradientImage::save(FileStorage& fs) const
{

}
void CGradientImage::load(FileStorage& fs)
{

}
