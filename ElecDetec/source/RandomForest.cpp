/*
 * RandomForest.cpp
 *
 *  Created on: Jul 8, 2014
 *      Author: test
 */

#include "RandomForest.h"

CRandomForest::CRandomForest(MODULE_CONSTRUCTOR_SIGNATURE)
{
	module_print_name_ = "RandomForest";
	is_trained_ = false;

	required_input_signature_ = DATA_TYPE_VECTOR | CV_32FC1; // takes float vector
	output_signature_ = DATA_TYPE_SCALAR | CV_32SC1;

	if(is_root)
		setAsRoot();

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
	max_depth = 8;
	min_sample_cnt = 3;
	max_num_of_trees_in_the_forest = 80;
	termcrit_type = CV_TERMCRIT_ITER; // CV_TERMCRIT_EPS;

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

CVisionData* CRandomForest::exec()
{
//	if(data.back()->getType() != TYPE_VECTOR)
//		throw(VisionDataTypeException(data.back()->getType(), TYPE_VECTOR));
	CVisionData working_data = getConcatenatedDataAndClearBuffer();

	Mat result_scalar = Mat::zeros(1,2,CV_32FC1); // First Value: Label, second: weight
	result_scalar.at<float>(0,0) = static_cast<float>(rf_->predict(working_data.data()));

	// for now: use probability = 1:
	result_scalar.at<float>(0,1) = static_cast<float>(1.0);

	//if(class_result->val_ != 0)
	//cout << "prediction result is: " << class_result->val_ << endl;

	return new CVisionData(result_scalar, DATA_TYPE_SCALAR);
}

// train_data contains for each sample one CVisionData and train_labels a CVisionData-Label
void CRandomForest::train()
{
	// train_data contains for each sample a row
	CVisionData train_data = getConcatenatedDataAndClearBuffer();

//	cv::Mat train_data_mat = train_data.mat_; // generate cvMat without copying the data. need CV_32FC1 cv::Mat as train data
//	cv::Mat train_labels_mat(train_labels.vec_, false); // generate cvMat without copying the data. need CV_32SC1 as train labels

//	cout << "Matrix: " << train_data_mat.rows << "x" << train_data_mat.cols << endl;
//	cout << train_data_mat.at<float>(0,0) << endl;
//
//	cout << "Type of Train Data Mat: " << train_data.signature2str() << endl;
//	cout << " with size: " << train_data.data().rows << " x " << train_data.data().cols << endl;
//	cout << "Type of Train Label Mat: " << data_labels_->data().type() << " should be " << CV_32SC1 << endl;
//	cout << " with size: " << data_labels_->data().rows << " x " << data_labels_->data().cols << endl;
//	cout << flush;

	const Mat& varIdx = Mat(); // vector: selected feature subset (masks colums of train data)
	const Mat& sampleIdx = Mat(); // vector: selected sample subset (masks rows of train data)
	const Mat& varType = Mat(); // vector: for regression
	const Mat& missingDataMask = Mat(); // vector: identifies missing labels of train data

	cout << "Training RandomForest. Please be patient..." << flush;
	srand(time(NULL)); // to be sure
	rf_->train(train_data.data(), CV_ROW_SAMPLE, data_labels_->data(), varIdx, sampleIdx, varType, missingDataMask, *rf_params_);

	cout << " done. " << rf_->get_tree_count() << " trees trained." << endl << flush;
}

void CRandomForest::save(FileStorage& fs) const
{
//	cout << "saving RandomForest..." << endl << flush;
	stringstream config_name;
	config_name << CONFIG_NAME_RF << "-" << module_id_;
	rf_->write(*fs, config_name.str().c_str());
//	cout << "RandomForest saved." << endl;
}

void CRandomForest::load(FileStorage& fs)
{
	if(rf_)
	{
		rf_->clear();
		stringstream config_name;
		config_name << CONFIG_NAME_RF << "-" << module_id_;
		rf_->read(*fs, cvGetFileNodeByName(*fs, NULL, config_name.str().c_str()));
	}
}
