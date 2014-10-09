/*
 * OrientationFilter.cpp
 *
 *  Created on: Oct 6, 2014
 *      Author: test
 */

#include "OrientationFilter.h"

COrientationFilter::COrientationFilter(MODULE_CONSTRUCTOR_SIGNATURE) :
orientation_bin_idx_(ORIENTATION_DEFAULT_BIN_IDX),
n_orientation_bins_(ORIENTATION_DEFAULT_N_BINS)
{
	// takes single channel image as input and produces also an image
	MODULE_CTOR_INIT("Orientation Filter", DATA_TYPE_IMAGE | CV_8UC1, DATA_TYPE_IMAGE | CV_8UC1);

	vector<string> params_vec;
	bool is_number = true;
	if(!module_params.empty())
	{
		params_vec = splitStringByDelimiter(module_params, MODULE_PARAM_DELIMITER);
		for(vector<string>::const_iterator s = params_vec.begin(); s != params_vec.end(); ++s)
		{
			for(string::const_iterator k = s->begin(); k != s->end(); ++k)
				is_number = is_number && isdigit(*k);
		}
	}

	// exactly 2 integers are required
	if(!is_number || params_vec.size() != 2)
	{
		cerr << "Orientation Filter parameters must be exactly two integer numbers: bin_idx (starting from 1) and #bins" << endl;
		exit(-1); // TODO: Throw exception?
	}

	for(unsigned int param_cnt = 0; param_cnt < params_vec.size(); ++param_cnt)
	{
		stringstream ss;
		ss << params_vec[param_cnt];
		switch(param_cnt)
		{
		case 0:
			ss >> orientation_bin_idx_;
			orientation_bin_idx_--;
			if(orientation_bin_idx_ < 0 || orientation_bin_idx_ >= n_orientation_bins_)
			{
				cerr << "Orientation Filter: invalid bin index: must be (1..n_bins)" << endl;
				exit(-1); // TODO: Throw exception?
			}
			break;
		case 1:
			ss >> n_orientation_bins_;
			break;
		}
	}

	// orientation to derive to x and y
	//float orientation = ((float)orientation_bin_idx_)*(M_PI/(float)n_orientation_bins_);
	// orientation of the bin center in radiant: IDX*STEP + STEP/2 = (IDX + 0.5)*STEP
	float orientation = ((float)orientation_bin_idx_+0.5)*(M_PI/(float)n_orientation_bins_);

	//cout << "Orientation is: " << orientation*180.0/M_PI << endl;
	x_weight_ = cos(orientation);
	y_weight_ = sin(orientation);

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


CVisionData* COrientationFilter::exec()
{
	CVisionData working_data = getConcatenatedDataAndClearBuffer();

	// Step 1: get for each image pixel the gradient for each color channel
	Mat orientation_grad_resp;
	Mat grad_x, grad_y;

	// Blur
	Mat blurred_img;
	GaussianBlur(working_data.data(), blurred_img, Size(3,3), 0, 0, BORDER_DEFAULT);
	// Gradient X
	Sobel(blurred_img, grad_x, ddepth_, 1, 0, 3, scale_, delta_, BORDER_DEFAULT);
	// Gradient Y
	Sobel(blurred_img, grad_y, ddepth_, 0, 1, 3, scale_, delta_, BORDER_DEFAULT);

	threshold(grad_x, grad_x, to_zero_threshold_, 0, THRESH_TOZERO);
	threshold(grad_y, grad_y, to_zero_threshold_, 0, THRESH_TOZERO);

	// superposition of both gradients
	addWeighted(grad_x, x_weight_, grad_y, y_weight_, 0, orientation_grad_resp);

	//normalize(out_img->mat_, out_img->mat_, 0, 255, NORM_MINMAX);
	Mat out_img;
	orientation_grad_resp.convertTo(out_img, CV_8UC1);
//	imshow("orig", working_data.data());
//	PAUSE_AND_SHOW(out_img);
	return new CVisionData(out_img, DATA_TYPE_IMAGE);
}

void COrientationFilter::save(FileStorage& fs) const
{
	stringstream config_name;
	config_name << ORIENTATION_CONFIG_NAME << "-" << module_id_;
	fs << config_name.str().c_str() << "{";
	fs << ORIENTATION_CONFIG_NAME_BIN_IDX << orientation_bin_idx_;
	fs << ORIENTATION_CONFIG_NAME_N_BINS << n_orientation_bins_;
	fs << "}";
}

void COrientationFilter::load(FileStorage& fs)
{
	stringstream config_name;
	config_name << ORIENTATION_CONFIG_NAME << "-" << module_id_;
	FileNode fn = fs[config_name.str().c_str()];
	fn[ORIENTATION_CONFIG_NAME_BIN_IDX] >> orientation_bin_idx_;
	fn[ORIENTATION_CONFIG_NAME_N_BINS] >> n_orientation_bins_;

	// orientation of the bin center in radiant: IDX*STEP + STEP/2 = (IDX + 0.5)*STEP
	float orientation = ((float)orientation_bin_idx_+0.5)*(M_PI/(float)n_orientation_bins_);

	//cout << "Orientation is: " << orientation*180.0/M_PI << endl;
	x_weight_ = cos(orientation);
	y_weight_ = sin(orientation);
}
