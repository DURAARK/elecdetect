/*
 * QGradient.cpp
 *
 *  Created on: Okt 6, 2014
 *      Author: test
 */


#include "QGradient.h"

CQuantizedGradient::CQuantizedGradient(MODULE_CONSTRUCTOR_SIGNATURE)
{
	module_print_name_ = "Quantized Gradient";
	required_input_signature_ = DATA_TYPE_IMAGE | CV_8UC3; // takes color image as input
	output_signature_ = DATA_TYPE_IMAGE | CV_8UC1;

	if(is_root)
		setAsRoot();

	n_bins_ = 5; // Number of quantization bins
	n_neigbours_ = 3; // NNxNN neighbourhood consideration for noise robustness. MUST BE ODD!
	mag_threshold_ = 20; // Threshold from which gradients are considered

	bin_step_ = M_PI/n_bins_; // precalc the quantization bin steps

	// Sobel parameters
	scale_ = 1;
	delta_ = 0;
	ddepth_ = CV_32F;
}

CQuantizedGradient::~CQuantizedGradient()
{

}

CVisionData* CQuantizedGradient::exec()
{
	CVisionData working_data = getConcatenatedDataAndClearBuffer();

	Mat color_channels[3];
	split(working_data.data(), color_channels);


	// Step 1: get for each image pixel the gradient for each color channel
	Mat channel_mag_resp[3];
	Mat grad_x[3], grad_y[3];
	for(int ccnt = 0; ccnt < 3; ++ccnt)
	{
		Mat blurred_img = color_channels[ccnt];

		// Blur
		//GaussianBlur(color_channels[ccnt], blurred_img, Size(3,3), 0, 0, BORDER_DEFAULT);
		// Gradient X
		Sobel(blurred_img, grad_x[ccnt], ddepth_, 1, 0, 3, scale_, delta_, BORDER_DEFAULT);
		// Gradient Y
		Sobel(blurred_img, grad_y[ccnt], ddepth_, 0, 1, 3, scale_, delta_, BORDER_DEFAULT);

		// superposition of the absolute of both directions
		magnitude(grad_x[ccnt], grad_y[ccnt], channel_mag_resp[ccnt]);
		//addWeighted(abs(grad_x[ccnt]), 0.5, abs(grad_y[ccnt]), 0.5, 0, channel_grad_resp[ccnt]);
	}

	// Step 2: for each pixel, quantize the largest gradient of the color channels
	Mat quantized_orientation_map = Mat::zeros(working_data.data().size(), CV_8UC1);
	Mat max_mag_channel_idx = Mat::zeros(working_data.data().size(), CV_8UC1);

	Mat max_grad_resp = Mat::zeros(working_data.data().size(), CV_32FC1);
	for(int row_cnt = 0; row_cnt < max_mag_channel_idx.rows; ++row_cnt)
	{
		for(int col_cnt = 0; col_cnt < max_mag_channel_idx.cols; ++col_cnt)
		{
			// for each point in the image
			Point2i pt(col_cnt, row_cnt);

			// search for RGB channel with the largest magnitude
			float max_val = channel_mag_resp[0].at<float>(pt);
			for(uchar ccnt = 1; ccnt < 3; ++ccnt)
			{
				if(channel_mag_resp[ccnt].at<float>(pt) > max_val)
				{
					max_mag_channel_idx.at<uchar>(pt) = ccnt;
					max_val = channel_mag_resp[ccnt].at<float>(pt);
				}
			}
			max_grad_resp.at<float>(pt) = max_val;

			// quantize the gradient (just take the orientations, i.e. (0-PI] )
//			float x_component = grad_x[max_mag_channel_idx.at<uchar>(pt)].at<float>(pt);
//			float y_component = grad_y[max_mag_channel_idx.at<uchar>(pt)].at<float>(pt);
//
//			float orientation = fmod(atan2(y_component, x_component)+2*M_PI, 180.0);
//
//			int bin = floor(orientation/bin_step_);
//			quantized_orientation_map.at<uchar>(pt) = bin;
		}
	}


	threshold(max_grad_resp, max_grad_resp, mag_threshold_, 0, THRESH_TOZERO);
	//Mat max_grad = max(max(color_grad_resp[0], color_grad_resp[1]), color_grad_resp[2]);





	// Step 3: assign majority quantized gradient value of the neighbourhood



	//normalize(out_img->mat_, out_img->mat_, 0, 255, NORM_MINMAX);
	Mat out_img;
	max_grad_resp.convertTo(out_img, CV_8UC1);
//	imshow("orig", working_data.data());
//	PAUSE_AND_SHOW(out_img);
	return new CVisionData(out_img, DATA_TYPE_IMAGE);

}

void CQuantizedGradient::save(FileStorage& fs) const
{

}
void CQuantizedGradient::load(FileStorage& fs)
{

}
