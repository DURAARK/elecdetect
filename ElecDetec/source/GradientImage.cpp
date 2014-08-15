/*
 * GradientImage.cpp
 *
 *  Created on: Jun 30, 2014
 *      Author: test
 */

#include "GradientImage.h"

CGradientImage::CGradientImage()
{
	module_print_name_ = "Gradient";

	scale_ = 3;
	delta_ = 0;
	ddepth_ = CV_16S;
}

CGradientImage::~CGradientImage()
{

}

void CGradientImage::exec(std::vector<CVisionData*>& data) throw(VisionDataTypeException)
{
	CMat* img0_ptr = (CMat*)data.back();
	CMat* out_img = new CMat();

	Mat img0_gray, grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	// Blur and Grayscale
	GaussianBlur(img0_ptr->mat_, img0_gray, Size(3,3), 0, 0, BORDER_DEFAULT);
	cvtColor(img0_gray, img0_gray, CV_RGB2GRAY);

	// Gradient X
	Sobel(img0_gray, grad_x, ddepth_, 1, 0, 3, scale_, delta_, BORDER_DEFAULT);
	// Gradient Y
	Sobel(img0_gray, grad_y, ddepth_, 0, 1, 3, scale_, delta_, BORDER_DEFAULT);

	// Conversion to absolute gradient image
	convertScaleAbs(grad_x, abs_grad_x );
	convertScaleAbs(grad_y, abs_grad_y );

	// superposition of both directions
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, out_img->mat_);

	//normalize(out_img->mat_, out_img->mat_, 0, 255, NORM_MINMAX);

	data.push_back(out_img);
}

void CGradientImage::save(FileStorage& fs) const
{

}
void CGradientImage::load(FileStorage& fs)
{

}
