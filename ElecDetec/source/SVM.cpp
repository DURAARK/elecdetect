/*
 * SVM.cpp
 *
 *  Created on: Jun 17, 2014
 *      Author: test
 */

#include "SVM.h"

CSVM::CSVM()
{
	module_name_ = "SVM";

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
	//nu = 0.5;
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

void CSVM::exec(std::vector<CVisionData*>& data) throw(VisionDataTypeException)
{
	if(data.back()->getType() != TYPE_VECTOR)
		throw(VisionDataTypeException(data.back()->getType(), TYPE_VECTOR));

	const vector<float> in_vec = (((CVector<float>*)data.back())->vec_);

	CWeightedScalar<int>* class_result = new CWeightedScalar<int>(static_cast<int>(svm_->predict(Mat(in_vec))));
	//if(class_result->val_ != 0)
	//	cout << "prediction result is: " << class_result->val_ << endl;

	data.push_back(class_result);
}


void CSVM::train(const CMat& train_data, const CVector<int>& train_labels) throw(VisionDataTypeException)
{
	// train_data contains for each sample a row

	cv::Mat train_data_mat = train_data.mat_; // generate cvMat without copying the data. need CV_32FC1 cv::Mat as train data
	cv::Mat train_labels_mat(train_labels.vec_, false); // generate cvMat without copying the data. need CV_32SC1 as train labels

	//cout << "Matrix: " << train_data_mat.rows << "x" << train_data_mat.cols << endl;
	//cout << train_data_mat.at<float>(0,0) << endl;
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
	svm_->train_auto(train_data_mat, train_labels_mat, Mat(), Mat(), *svm_params_, 5);

	cout << " done. Support Vectors: " << svm_->get_support_vector_count() << endl << flush;

}

void CSVM::save(FileStorage& fs) const
{
//	cout << "saving SVM..." << endl << flush;
	svm_->write(*fs, CONFIG_NAME_SVM);
//	cout << "SVM saved." << endl;
}

void CSVM::load(FileStorage& fs)
{
	if(svm_)
	{
		svm_->clear();
		svm_->read(*fs, cvGetFileNodeByName(*fs, NULL, CONFIG_NAME_SVM));
	}
}
