/*
 * SVM.cpp
 *
 *  Created on: Jun 17, 2014
 *      Author: test
 */

#include "SVM.h"

CSVM::CSVM(MODULE_CONSTRUCTOR_SIGNATURE)
{
	module_print_name_ = "SVM";
	is_trained_ = false;

	required_input_signature_ = DATA_TYPE_VECTOR | CV_32FC1; // takes float vector
	output_signature_ = DATA_TYPE_WEIGHTED_SCALAR | CV_32FC1;

	if(is_root)
		setAsRoot();

	// SVM Params
	// int svm_type, int kernel_type, double degree, double gamma, double coef0, double Cvalue, double nu, double p, CvMat* class_weights, CvTermCriteria term_crit
	// default:     svm_type(CvSVM::C_SVC), kernel_type(CvSVM::RBF), degree(0), gamma(1), coef0(0), C(1), nu(0), p(0), class_weights(0)

	//defaults:
	int svm_type = CvSVM::C_SVC, kernel_type = CvSVM::RBF;
	double degree = 0, gamma = 1, coef0 = 0, Cvalue = 1, nu = 0, p = 0;
	CvMat* class_weights = NULL;
	// Default: TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
	TermCriteria term_crit(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, FLT_EPSILON);

	// own config:
	//svm_type = CvSVM::NU_SVC;
	//nu = 0.8;
	//Cvalue = 0.4;
	//test

	svm_params_ = new SVMParams(svm_type, kernel_type, degree, gamma, coef0, Cvalue, nu, p, class_weights, term_crit);
	svm_ = new CvSVM();
}

CSVM::~CSVM()
{
	if(svm_)
		delete svm_;
	svm_ = NULL;

	if(svm_params_)
		delete svm_params_;
	svm_params_ = NULL;
}

CVisionData* CSVM::exec()
{
	// data-buffer already stores the converted data
	CVisionData* working_data = getConcatenatedDataAndClearBuffer();

	Mat result_scalar = Mat::zeros(1,2,CV_32FC1); // First Value: Label, second: weight
	result_scalar.at<float>(0,0) = static_cast<float>(svm_->predict(working_data->data()));
	//if(class_result->val_ != 0)
	//	cout << "prediction result is: " << class_result->val_ << endl;

	return new CVisionData(result_scalar, DATA_TYPE_WEIGHTED_SCALAR);
}


void CSVM::train()
{
	// train_data contains for each sample a row
	//assert(train_data.data().rows == static_cast<int>(train_labels.data().rows));
	CVisionData* train_data = getConcatenatedDataAndClearBuffer();

	//cv::Mat train_data_mat = train_data.mat_; // generate cvMat without copying the data. need CV_32FC1 cv::Mat as train data
	//cv::Mat train_labels_mat(train_labels.vec_, false); // generate cvMat without copying the data. need CV_32SC1 as train labels

//	cout << "Matrix: " << train_data_mat.rows << "x" << train_data_mat.cols << endl;
//	cout << "first value" << train_data_mat.at<float>(0,0) << endl;

	double min, max;
	minMaxLoc(train_data->data(), &min, &max);
	cout << "min-max: " << min << " - " << max << endl << flush;

//	// constructor for matrix headers pointing to user-allocated data
//    Mat(int _rows, int _cols, int _type, void* _data, size_t _step=AUTO_STEP);
//    Mat(Size _size, int _type, void* _data, size_t _step=AUTO_STEP);

//	cout << "Type of Train Data Mat: " << train_data.type2str() << endl;
//	cout << " with size: " << train_data_mat.rows << " x " << train_data_mat.cols << endl;
//	cout << "Type of Train Label Mat: " << train_labels_mat.type() << " should be " << CV_32SC1 << endl;
//	cout << " with size: " << train_labels_mat.rows << " x " << train_labels_mat.cols << endl;

//	cout << flush;

	cout << "Training SVM. Please be patient..." << flush;

	//svm_->train(train_data_mat, train_labels_mat, Mat(), Mat(), *svm_params_);
	svm_->train_auto(train_data->data(), data_labels_->data(), Mat(), Mat(), *svm_params_, 5);

	cout << " done. Support Vectors: " << svm_->get_support_vector_count() << endl << flush;
}

void CSVM::save(FileStorage& fs) const
{
//	cout << "saving SVM..." << endl << flush;
	stringstream config_name;
	config_name << CONFIG_NAME_SVM << "-" << module_id_;
	svm_->write(*fs, config_name.str().c_str());
//	cout << "SVM saved." << endl;
}

void CSVM::load(FileStorage& fs)
{
	if(svm_)
	{
		stringstream config_name;
		config_name << CONFIG_NAME_SVM << "-" << module_id_;
		svm_->clear();
		svm_->read(*fs, cvGetFileNodeByName(*fs, NULL, config_name.str().c_str()));
	}
}
