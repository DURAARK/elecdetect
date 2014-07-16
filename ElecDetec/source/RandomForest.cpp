/*
 * RandomForest.cpp
 *
 *  Created on: Jul 8, 2014
 *      Author: test
 */

#include "RandomForest.h"

CRandomForest::CRandomForest()
{
	module_name_ = "RandomForest";

	//Default values:
	int max_depth = 5;
	int min_sample_cnt = 10;
	float regression_arruracy = 0;
	bool use_surrogates = false;
	int max_categories = 10;
	const float* priors = 0;
	bool calc_var_importance = false;
	int nactive_vars = 0;
	int max_num_of_trees_in_the_forest = 50;
	float forest_accuracy = 0.1;
	int termcrit_type = CV_TERMCRIT_ITER + CV_TERMCRIT_EPS;

	// own configuration:
	max_depth = 10;
	min_sample_cnt = 2;
	max_num_of_trees_in_the_forest = 100;
	termcrit_type = CV_TERMCRIT_ITER;

	rf_params_ = new CvRTParams(max_depth, min_sample_cnt, regression_arruracy, use_surrogates,
			                    max_categories, priors, calc_var_importance, nactive_vars,
			                    max_num_of_trees_in_the_forest, forest_accuracy, termcrit_type);

	rf_ = new CvRTrees();
}

CRandomForest::~CRandomForest()
{
	if(rf_)
		delete rf_;
	rf_ = NULL;

	if(rf_params_)
		delete rf_params_;
	rf_params_ = NULL;
}

void CRandomForest::exec(std::vector<CVisionData*>& data) throw(VisionDataTypeException)
{
	if(data.back()->getType() != TYPE_VECTOR)
		throw(VisionDataTypeException(data.back()->getType(), TYPE_VECTOR));

	const vector<float> in_vec = (((CVector<float>*)data.back())->vec_);

	CWeightedScalar<int>* class_result = new CWeightedScalar<int>(static_cast<int>(rf_->predict(Mat(in_vec))));
	//if(class_result->val_ != 0)
	//cout << "prediction result is: " << class_result->val_ << endl;

	data.push_back(class_result);
}

// train_data contains for each sample one CVisionData and train_labels a CVisionData-Label
void CRandomForest::train(const CMat& train_data, const CVector<int>& train_labels) throw(VisionDataTypeException)
{
	// train_data contains for each sample a row
	cv::Mat train_data_mat = train_data.mat_; // generate cvMat without copying the data. need CV_32FC1 cv::Mat as train data
	cv::Mat train_labels_mat(train_labels.vec_, false); // generate cvMat without copying the data. need CV_32SC1 as train labels

//	cout << "Matrix: " << train_data_mat.rows << "x" << train_data_mat.cols << endl;
//	cout << train_data_mat.at<float>(0,0) << endl;
//
//	cout << "Type of Train Data Mat: " << train_data.type2str() << endl;
//	cout << " with size: " << train_data_mat.rows << " x " << train_data_mat.cols << endl;
//	cout << "Type of Train Label Mat: " << train_labels_mat.type() << " should be " << CV_32SC1 << endl;
//	cout << " with size: " << train_labels_mat.rows << " x " << train_labels_mat.cols << endl;
//	cout << flush;

	const Mat& varIdx = Mat(); // vector: selected feature subset (masks colums of train data)
	const Mat& sampleIdx = Mat(); // vector: selected sample subset (masks rows of train data)
	const Mat& varType = Mat(); // vector: for regression
	const Mat& missingDataMask = Mat(); // vector: identifies missing labels of train data

	cout << "Training RandomForest. Please be patient..." << endl;
	srand(time(NULL)); // to be sure
	rf_->train(train_data_mat, CV_ROW_SAMPLE, train_labels_mat, varIdx, sampleIdx, varType, missingDataMask, *rf_params_);

	cout << "RandomForest training completed! " << rf_->get_tree_count() << " trees trained." << endl;
}

void CRandomForest::save(FileStorage& fs) const
{
//	cout << "saving RandomForest..." << endl << flush;
	rf_->write(*fs, CONFIG_NAME_RF);
//	cout << "RandomForest saved." << endl;
}

void CRandomForest::load(FileStorage& fs)
{
	if(rf_)
	{
		rf_->clear();
		rf_->read(*fs, cvGetFileNodeByName(*fs, NULL, CONFIG_NAME_RF));
	}
}
