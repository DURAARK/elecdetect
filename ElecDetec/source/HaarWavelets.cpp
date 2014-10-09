/*
 * HaarWavelets.cpp
 *
 *  Created on: Oct 6, 2014
 *      Author: test
 */

#include "HaarWavelets.h"

CHaarWavelets::CHaarWavelets(MODULE_CONSTRUCTOR_SIGNATURE) :
n_tests_(HAAR_FEAT_DEFAULT_N_TESTS),
symm_percentage_(HAAR_FEAT_DEFAULT_SYM_PERCENT)
{
	// takes single channel image as input and produces a float vector
	MODULE_CTOR_INIT("Haar-like Wavelets", DATA_TYPE_IMAGE | CV_8UC1, DATA_TYPE_VECTOR | CV_32FC1);

	// Notes on the input type "float image": the pair-wise test is a cosine-similarity measure, i.e. if
	// the image-pixels are radiants, 0 and 2*pi are considered as similar. When passing a 8 or 32 bit image,
	// the data converter automatically converts the data to [0..1] range, where the cosine is more or less linear.

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

	initWavelets();
}

CHaarWavelets::~CHaarWavelets()
{

}


void CHaarWavelets::initWavelets()
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


CVisionData* CHaarWavelets::exec()
{
	CVisionData working_data = getConcatenatedDataAndClearBuffer();
	vector<float> haar_features;

	Mat integral_img;
	integral(working_data.data(), integral_img, CV_32S);

	for(vector<RectPair>::const_iterator pair_it = wavelet_pairs_.begin(); pair_it != wavelet_pairs_.end(); ++pair_it)
	{
		float feature;

		const Point tl1 = pair_it->first.tl();
		const Point tr1(pair_it->first.x + pair_it->first.width, pair_it->first.y);
		const Point br1 = pair_it->first.br();
		const Point bl1(pair_it->first.x, pair_it->first.y + pair_it->first.height);
		const int rect1_sum = integral_img.at<int>(br1) - integral_img.at<int>(bl1) - integral_img.at<int>(tr1) + integral_img.at<int>(tl1);
		const float rect1_mean = (float)rect1_sum/pair_it->first.area();

		if(pair_it->second.area()) // empty second rect: only the first rect is used!
		{
			const Point tl2 = pair_it->second.tl();
			const Point tr2(pair_it->second.x + pair_it->second.width, pair_it->second.y);
			const Point br2 = pair_it->second.br();
			const Point bl2(pair_it->second.x, pair_it->second.y + pair_it->second.height);
			const int rect2_sum = integral_img.at<int>(br2) - integral_img.at<int>(bl2) - integral_img.at<int>(tr2) + integral_img.at<int>(tl2);
			const float rect2_mean = (float)rect2_sum/pair_it->second.area();
			feature = fabs(rect1_mean - rect2_mean)/255.0;
		}
		else
		{
			feature = rect1_mean/255.0;
		}
		haar_features.push_back(feature);
	}

	return new CVisionData(Mat(haar_features).reshape(0,1).clone(), DATA_TYPE_VECTOR);
}

void CHaarWavelets::save(FileStorage& fs) const
{
	stringstream config_name;
	config_name << HAAR_FEAT_CONFIG_NAME << "-" << module_id_;
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

void CHaarWavelets::load(FileStorage& fs)
{
	wavelet_pairs_.clear();
	stringstream config_name;
	config_name << HAAR_FEAT_CONFIG_NAME << "-" << module_id_;

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
