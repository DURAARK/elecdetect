///*
// * DummyFeature.cpp
// *
// *  Created on: Jul 4, 2014
// *      Author: test
// */
//
//#include "DummyFeature.h"
//
//CDummyFeature::CDummyFeature()
//{
//	module_print_name_ = "Dummy";
//
//	canonical_size_ = Size(100,100);
//	feature_length_ = canonical_size_.area();
//}
//
//CDummyFeature::~CDummyFeature()
//{
//
//}
//
//void CDummyFeature::exec(std::vector<CVisionData*>& data) throw(VisionDataTypeException)
//{
//	if(data.back()->getType() != TYPE_MAT)
//		throw(VisionDataTypeException(data.back()->getType(), TYPE_MAT));
//
//	const cv::Mat img0 = (((CMat*)data.back())->mat_);
//
//	// grayscale image and resize input image to match feature size
//	cv::Mat img_gray;
//	if (img0.channels() != 1)
//		cv::cvtColor(img0, img_gray, cv::COLOR_BGR2GRAY);
//	else
//		img_gray = img0;
//
//	if(img_gray.size() != canonical_size_)
//		cv::resize(img_gray, img_gray, canonical_size_);
//
//	//cout << img0.rows << " x " << img0.cols << endl;
//
//	CVector<float>* features = new CVector<float>();
//	Mat img_gray_f;
//	img_gray.convertTo(img_gray_f, CV_32FC1, 1/255.0);
//	//normalize(img_gray_f, img_gray_f, 0.0, 1.0);
//	//cout << img_gray_f.rows << " x " << img_gray_f.cols << endl;
//	//cout << CMat(img_gray_f).type2str() << endl;
//	features->vec_.assign((float*)img_gray_f.datastart, (float*)img_gray_f.dataend);
//	//cout << "Feature Size: " << features->vec_.size() << endl;
//
//	data.push_back(features);
//}
//
//void CDummyFeature::save(FileStorage& fs) const
//{
//
//}
//void CDummyFeature::load(FileStorage& fs)
//{
//
//}
//
//int CDummyFeature::getFeatureLength()
//{
//	return feature_length_;
//}
