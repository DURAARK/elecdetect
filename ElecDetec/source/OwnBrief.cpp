/*
 * Brief.cpp
 *
 *  Created on: Jul 7, 2014
 *      Author: test
 */

#include "OwnBrief.h"

COwnBrief::COwnBrief(int inchain_input_signature) : feature_length_(DEFAULT_FEATURE_LENGTH)
{
	module_print_name_ = "Brief";
	
	required_input_signature_mask_ = DATA_TYPE_IMAGE | DATA_TYPE_VECTOR | CV_32FC1; // needs float image or vector as input
	output_type_ = DATA_TYPE_VECTOR | CV_32FC1;

	if(!(inchain_input_signature & required_input_signature_mask_))
	{
		// just perform data format conversions
		if(inchain_input_signature & DATA_TYPE_IMAGE)
			data_converter_ = new CDataConverter(inchain_input_signature, DATA_TYPE_IMAGE | CV_32FC1);
		else if(inchain_input_signature & DATA_TYPE_VECTOR)
			data_converter_ = new CDataConverter(inchain_input_signature, DATA_TYPE_VECTOR | CV_32FC1);
		else // otherwise wait for exception
			data_converter_ = new CDataConverter(inchain_input_signature, DATA_TYPE_IMAGE | CV_32FC1);
	}

	initTestPairs();
}

COwnBrief::~COwnBrief()
{
	// nothing to clear here
}

void COwnBrief::initTestPairs()
{
	rel_test_pairs_.clear();
	srand(time(NULL));
	for(int test_cnt = 0; test_cnt < feature_length_; ++test_cnt)
	{
		Point2f pt1(static_cast<float>(rand()) / static_cast<float>(RAND_MAX),
				    static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
		Point2f pt2(static_cast<float>(rand()) / static_cast<float>(RAND_MAX),
					static_cast<float>(rand()) / static_cast<float>(RAND_MAX));

		test_pair pt_pair(pt1, pt2);
		rel_test_pairs_.push_back(pt_pair);
	}
}

float COwnBrief::compare(const Mat& img0, const Point2i& pt1, const Point2i& pt2) const
{
	return img0.at<float>(pt1) - img0.at<float>(pt2);
}


void COwnBrief::exec(const CVisionData& input_data, CVisionData& output_data)
{
	CVisionData working_data(input_data.data(), input_data.getType());
	if(data_converter_)
	{
		data_converter_->convert(working_data);
	}
	vector<float> brief_features;

//	cout << img_gray_float;
//	cout << "Should be CV_32FC1: " << CMat(img_gray_float).type2str() << endl;

	int input_img_cols = working_data.data().cols;
	int input_img_rows = working_data.data().rows;

	vector<test_pair>::const_iterator test_it;
	for(test_it = rel_test_pairs_.begin(); test_it != rel_test_pairs_.end(); ++ test_it)
	{
		const Point2i abs_pt1((*test_it).first.x*input_img_cols, (*test_it).first.y*input_img_rows);
		const Point2i abs_pt2((*test_it).second.x*input_img_cols, (*test_it).second.y*input_img_rows);
//		cout << "Testing Point: " << abs_pt1 << " against Point: " << abs_pt2 <<
//				" (orig: " << (*test_it).first << " and " << (*test_it).second << endl << flush;
		brief_features.push_back(compare(working_data.data(), abs_pt1, abs_pt2));
	}

	output_data.assignData(Mat(brief_features), DATA_TYPE_VECTOR);
}

void COwnBrief::save(FileStorage& fs) const
{
	stringstream config_name;
	config_name << CONFIG_NAME_TESTPAIRS << "-" << module_id_;
	//fs << CONFIG_NAME_TESTPAIRS << rel_test_pairs_;
    fs << config_name.str().c_str() << "[";
	vector<test_pair>::const_iterator t_it;
	for(t_it = rel_test_pairs_.begin(); t_it != rel_test_pairs_.end(); ++ t_it)
    {
        fs << "{:" << "pt1" << t_it->first << "pt2" << t_it->second;
        fs << "}";
    }
    fs << "]";
}

void COwnBrief::load(FileStorage& fs)
{
	rel_test_pairs_.clear();
	stringstream config_name;
	config_name << CONFIG_NAME_TESTPAIRS << "-" << module_id_;

	FileNode config_tests = fs[config_name.str().c_str()];
	for(FileNodeIterator t_it = config_tests.begin(); t_it != config_tests.end(); ++t_it)
	{
		Point2f pt1; (*t_it)["pt1"] >> pt1;
		Point2f pt2; (*t_it)["pt2"] >> pt2;
		rel_test_pairs_.push_back(test_pair(pt1, pt2));
	}
	feature_length_ = rel_test_pairs_.size();
}



