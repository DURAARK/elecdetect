/*
 * Brief.cpp
 *
 *  Created on: Jul 7, 2014
 *      Author: test
 */

#include "OwnBrief.h"

COwnBrief::COwnBrief() : feature_length_(DEFAULT_FEATURE_LENGTH)
{
	module_print_name_ = "Brief";

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


void COwnBrief::exec(std::vector<CVisionData*>& data) throw(VisionDataTypeException)
{
	// Caclulate gray, normalized to 1 image CV_32FC1
	if(data.back()->getType() != TYPE_MAT)
			throw(VisionDataTypeException(data.back()->getType(), TYPE_MAT));

	const Mat img0 = ((CMat*)data.back())->mat_;

	CVector<float>* brief_features = new CVector<float>();

	Mat img_gray, img_gray_float;
	if (img0.channels() != 1)
		cv::cvtColor(img0, img_gray, cv::COLOR_BGR2GRAY);
	else
		img_gray = img0;

	img_gray.convertTo(img_gray_float, CV_32FC1, 1/255.0);
//	cout << img_gray_float;
//	cout << "Should be CV_32FC1: " << CMat(img_gray_float).type2str() << endl;

	vector<test_pair>::const_iterator test_it;
	for(test_it = rel_test_pairs_.begin(); test_it != rel_test_pairs_.end(); ++ test_it)
	{
		const Point2i abs_pt1((*test_it).first.x*img0.cols, (*test_it).first.y*img0.rows);
		const Point2i abs_pt2((*test_it).second.x*img0.cols, (*test_it).second.y*img0.rows);
//		cout << "Testing Point: " << abs_pt1 << " against Point: " << abs_pt2 <<
//				" (orig: " << (*test_it).first << " and " << (*test_it).second << endl << flush;
		brief_features->vec_.push_back(compare(img_gray_float, abs_pt1, abs_pt2));
	}

	data.push_back(brief_features);

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



