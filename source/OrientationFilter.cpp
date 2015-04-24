/*
 * ElecDetec: OrientationFilter.cpp
 *
 *  Created on: Feb, 2015
 *      Author: Robert Viehauser
 */

#include "OrientationFilter.h"

COrientationFilter::COrientationFilter(const Direction& direction) :
    direction_(direction)
{
    // parameters:
    // direction: enum type Direction that specifies if the filter responds to horizontal or vertical gradients

	// Threshold to reduce noise
	to_zero_threshold_ = 30.0;

	// Sobel parameters
	scale_ = 1;
	delta_ = 0;
	ddepth_ = CV_32F;
}

COrientationFilter::~COrientationFilter()
{

}


void COrientationFilter::filterImage(const Mat& input_img, Mat& filtered_img)
{
    // Step 1: get for each image pixel the gradient for each color channel
    Mat orientation_grad_resp;
    Mat grad_x, grad_y;

    // Blur
    Mat blurred_img;
    GaussianBlur(input_img, blurred_img, Size(3,3), 0, 0, BORDER_DEFAULT);
    // Gradient X (right)
    Sobel(blurred_img, grad_x, ddepth_, 1, 0, 3, scale_, delta_, BORDER_DEFAULT);
    // Gradient Y (down)
    Sobel(blurred_img, grad_y, ddepth_, 0, 1, 3, scale_, delta_, BORDER_DEFAULT);

    grad_x = abs(grad_x);
    grad_y = abs(grad_y);

    // threshold to zero to neglect gradients in opposite directions
    threshold(grad_x, grad_x, to_zero_threshold_, 0, THRESH_TOZERO);
    threshold(grad_y, grad_y, to_zero_threshold_, 0, THRESH_TOZERO);

    // superposition of both gradients: neglect gradients that are also strong to the perpenticular direction
    switch(direction_)
    {
    case HORIZ:
        addWeighted(grad_x, -0.707107, grad_y, 0.707107, 0, orientation_grad_resp);
        break;
    case VERT:
        addWeighted(grad_x, 0.707107, grad_y, -0.707107, 0, orientation_grad_resp);
        break;
    }

    //normalize(orientation_grad_resp, orientation_grad_resp, 0, 255, NORM_MINMAX);
    filtered_img = Mat::zeros(orientation_grad_resp.size(), CV_8UC1);
    orientation_grad_resp.convertTo(filtered_img, CV_8UC1);
}

