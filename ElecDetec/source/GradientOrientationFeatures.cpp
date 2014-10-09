/*
 * GradientOrientationFeatures.cpp
 *
 *  Created on: Oct 9, 2014
 *      Author: test
 */

#include "GradientOrientationFeatures.h"

CGradientOrientationFeatures::CGradientOrientationFeatures(MODULE_CONSTRUCTOR_SIGNATURE) :
n_tests_(GOF_DEFAULT_N_TESTS),
symm_percentage_(GOF_DEFAULT_SYM_PERCENT)
{
	// takes single channel image as input and produces a float vector
	MODULE_CTOR_INIT("Gradient Orientation Features", DATA_TYPE_IMAGE | CV_8UC3, DATA_TYPE_VECTOR | CV_32FC1);

	// the pair-wise test is a cosine-similarity measure. Only the orientation is considered, i.e. if
	// the image-pixels are radiants, 0 and pi are considered as similar.
	// As orientation of one pixel, the gradient with the highest magnitude of the 3 color channel is
	// assigned

	// read module parameters:
	// first param (int): number of testpairs
	// second param (int): percentage of symmetric testpairs
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

	// every parameter must be a number
	if(!is_number)
	{
		cerr << "Haar-Like Feature parameters must be an integer number" << endl;
		exit(-1);
	}

	for(unsigned int param_cnt = 0; param_cnt < params_vec.size(); ++param_cnt)
	{
		stringstream ss;
		ss << params_vec[param_cnt];
		switch(param_cnt)
		{
		case 0:
			ss >> n_tests_;
			break;
		case 1:
			ss >> symm_percentage_;
			break;
		}
	}

	// Threshold to reduce noise
	mag_threshold_ = 30.0;

	// Sobel parameter
	ddepth_ = CV_32F;

}

CGradientOrientationFeatures::~CGradientOrientationFeatures()
{

}

void CGradientOrientationFeatures::initWavelets()
{
	const int pair_percentage = 50; // percentage of pair-tests (vs. single rects)

	// generate test-boxes (relative coordinates within the image patch)
	for(int i = 0; i < n_tests_; ++i)
	{
		Rect rect1;
		rect1.width = randRange<int>(HAAR_FEAT_MIN_RECT_SZ, HAAR_FEAT_MAX_RECT_SZ);
		rect1.height = randRange<int>(HAAR_FEAT_MIN_RECT_SZ, HAAR_FEAT_MAX_RECT_SZ);
		rect1.x = randRange<int>(0, SWIN_SIZE-rect1.width-1);
		rect1.y = randRange<int>(0, SWIN_SIZE-rect1.height-1);

		Rect rect2;
		if(pair_percentage > randRange<int>(0,99))
		{
			if(symm_percentage_ && (symm_percentage_ > randRange<int>(0,99)))
			{
				// Percentage of the types of the symmetric arranged pairs:
				// Horizontal-symmetric, Vertical-symmetric, Center-symmetric
				const float percent_h_sym = 0.3;
				const float percent_v_sym = 0.3; // the rest is center-symmetric

				float sym_type = randRange<float>(0,1);

				rect2.height = rect1.height;
				rect2.width = rect1.width;
				if(sym_type < percent_h_sym) // h-symmetric
				{
					rect2.x = SWIN_SIZE - rect1.x - rect1.width; // see notes for derivation
					rect2.y = rect1.y;
				}
				else if(sym_type < percent_h_sym+percent_v_sym) // v-symmetric
				{
					rect2.x = rect1.x;
					rect2.y = SWIN_SIZE - rect1.y - rect1.height;
				}
				else // c-symmetric
				{
					rect2.x = SWIN_SIZE - rect1.x - rect1.width;
					rect2.y = SWIN_SIZE - rect1.y - rect1.height;
				}
			}
			else
			{
				rect2.width = randRange<int>(HAAR_FEAT_MIN_RECT_SZ, HAAR_FEAT_MAX_RECT_SZ);
				rect2.height = randRange<int>(HAAR_FEAT_MIN_RECT_SZ, HAAR_FEAT_MAX_RECT_SZ);
				rect2.x = randRange<int>(0, SWIN_SIZE-rect2.width-1);
				rect2.y = randRange<int>(0, SWIN_SIZE-rect2.height-1);
			}
		}
		wavelet_pairs_.push_back(RectPair(rect1, rect2));
	}

}

CVisionData* CGradientOrientationFeatures::exec()
{
	CVisionData working_data = getConcatenatedDataAndClearBuffer();

	Mat color_channels[3];
	split(working_data.data(), color_channels);

	// Step 1: get for each image pixel the gradient for each color channel
	Mat channel_mag_resp[3];
	Mat d_x[3], d_y[3];
	for(int ccnt = 0; ccnt < 3; ++ccnt)
	{
		Mat cur_channel = color_channels[ccnt];

		// Blur
		//GaussianBlur(color_channels[ccnt], blurred_img, Size(3,3), 0, 0, BORDER_DEFAULT);
		// Gradient X
		Sobel(cur_channel, d_x[ccnt], ddepth_, 1, 0, 3);
		// Gradient Y
		Sobel(cur_channel, d_y[ccnt], ddepth_, 0, 1, 3);

		// superposition of the absolute of both directions
		magnitude(d_x[ccnt], d_y[ccnt], channel_mag_resp[ccnt]);
	}

	// Step 2: for each pixel, find the largest gradient of the color channels
	Mat thresholded_dominant_mag = Mat::zeros(working_data.data().size(), CV_32FC1);
	Mat dominant_orientations = Mat::zeros(working_data.data().size(), CV_32FC1);
	for(int row_cnt = 0; row_cnt < dominant_orientations.rows; ++row_cnt)
	{
		for(int col_cnt = 0; col_cnt < dominant_orientations.cols; ++col_cnt)
		{
			// for each point in the image
			Point2i pt(col_cnt, row_cnt);
			int dominant_ch_idx = -1;

			// search for RGB channel with the largest magnitude > mag_threshold
			float max_val = mag_threshold_;
			for(uchar ccnt = 0; ccnt < 3; ++ccnt)
			{
				if(channel_mag_resp[ccnt].at<float>(pt) > max_val)
				{
					max_val = channel_mag_resp[ccnt].at<float>(pt);
					dominant_ch_idx = ccnt;
				}
			}
			if(dominant_ch_idx != -1)
			{
				thresholded_dominant_mag.at<float>(pt) = max_val;
				float dx = d_x[dominant_ch_idx].at<float>(pt);
				float dy = d_y[dominant_ch_idx].at<float>(pt);
				dominant_orientations.at<float>(pt) = fmod(atan2(dy,dx), M_PI); // only consider the orientation (0..pi)
			}
		}
	}

	// Dominant orientation visualization:
	Mat vis_img = Mat::zeros(working_data.data().size(), CV_8UC3);
	for(int row_cnt = 0; row_cnt < dominant_orientations.rows; ++row_cnt)
	{
		for(int col_cnt = 0; col_cnt < dominant_orientations.cols; ++col_cnt)
		{
			Point2i pt(col_cnt, row_cnt);
			if(thresholded_dominant_mag.at<float>(pt) != 0)
				vis_img.at<Vec3b>(pt) = Vec3b(dominant_orientations.at<float>(pt), 0, thresholded_dominant_mag.at<float>(pt));
		}
	}
	cvtColor(vis_img, vis_img, CV_HSV2BGR);
	PAUSE_AND_SHOW(vis_img);
	//

	// Calculate orientation Haar features:
	vector<float> haar_features;

	Mat pixel_mask = (thresholded_dominant_mag > 0)/255;
	Mat integral_orientation_img, integral_pixel_mask;
	integral(dominant_orientations, integral_orientation_img, CV_32F);
	integral(pixel_mask, integral_pixel_mask, CV_32S);

	for(vector<RectPair>::const_iterator pair_it = wavelet_pairs_.begin(); pair_it != wavelet_pairs_.end(); ++pair_it)
	{
		float feature;

		const Point tl1 = pair_it->first.tl();
		const Point tr1(pair_it->first.x + pair_it->first.width, pair_it->first.y);
		const Point br1 = pair_it->first.br();
		const Point bl1(pair_it->first.x, pair_it->first.y + pair_it->first.height);
		const float rect1_sum = integral_orientation_img.at<float>(br1) - integral_orientation_img.at<float>(bl1) - integral_orientation_img.at<float>(tr1) + integral_orientation_img.at<float>(tl1);
		const int n_relevant_px = integral_pixel_mask.at<int>(br1) - integral_pixel_mask.at<int>(bl1) - integral_pixel_mask.at<int>(tr1) + integral_pixel_mask.at<int>(tl1);
		const float rect1_mean = rect1_sum/n_relevant_px;

		if(pair_it->second.area()) // empty second rect: only the first rect is used!
		{
			const Point tl2 = pair_it->second.tl();
			const Point tr2(pair_it->second.x + pair_it->second.width, pair_it->second.y);
			const Point br2 = pair_it->second.br();
			const Point bl2(pair_it->second.x, pair_it->second.y + pair_it->second.height);
			const float rect2_sum = integral_orientation_img.at<float>(br2) - integral_orientation_img.at<float>(bl2) - integral_orientation_img.at<float>(tr2) + integral_orientation_img.at<float>(tl2);
			const int n_relevant_px = integral_pixel_mask.at<int>(br2) - integral_pixel_mask.at<int>(bl2) - integral_pixel_mask.at<int>(tr2) + integral_pixel_mask.at<int>(tl2);
			const float rect2_mean = rect2_sum/n_relevant_px;
			feature = fabs(rect1_mean - rect2_mean);
		}
		else
		{
			feature = rect1_mean;
		}
		haar_features.push_back(feature);
	}

	return new CVisionData(Mat(haar_features).reshape(0,1).clone(), DATA_TYPE_VECTOR);
}

void CGradientOrientationFeatures::save(FileStorage& fs) const
{
	stringstream config_name;
	config_name << GOF_CONFIG_NAME << "-" << module_id_;
    fs << config_name.str().c_str() << "{";
    fs << "n-tests" << n_tests_;
    fs << "symm-percent" << symm_percentage_;
    fs << "rect-pairs" << "[:";
	for(vector<RectPair>::const_iterator pair_it = wavelet_pairs_.begin(); pair_it != wavelet_pairs_.end(); ++pair_it)
    {
        fs << pair_it->first;
        fs << pair_it->second;
    }
	fs << "]";
    fs << "}";
}

void CGradientOrientationFeatures::load(FileStorage& fs)
{
	wavelet_pairs_.clear();
	stringstream config_name;
	config_name << GOF_CONFIG_NAME << "-" << module_id_;

	FileNode mod_node = fs[config_name.str().c_str()];
	mod_node["n-tests"] >> n_tests_;
	mod_node["symm-percent"] >> symm_percentage_;

	FileNode rect_pairs_node = mod_node["rect-pairs"];
	for(FileNodeIterator pair_node_it = rect_pairs_node.begin(); pair_node_it != rect_pairs_node.end(); ++pair_node_it)
	{
		Rect rect1, rect2;
		(*pair_node_it) >> rect1;
		pair_node_it++;
		(*pair_node_it) >> rect2;
		wavelet_pairs_.push_back(RectPair(rect1, rect2));
	}
}
