/*
 * PCA.cpp
 *
 *  Created on: Aug 4, 2014
 *      Author: test
 */

#include "PCA.h"

CPCA::CPCA(MODULE_CONSTRUCTOR_SIGNATURE) : n_eigenvectors_(DEFAULT_NUMBER_OF_EIGENVECTORS), opencv_pca_ptr_(NULL)
{
	module_print_name_ = "PCA";
	is_trained_ = false;
	required_input_signature_ = DATA_TYPE_VECTOR | CV_32FC1; // takes float vector
	output_signature_ = DATA_TYPE_VECTOR | CV_32FC1;

	if(is_root)
		setAsRoot();
}

CPCA::~CPCA()
{
	if(opencv_pca_ptr_)
		delete opencv_pca_ptr_;

	opencv_pca_ptr_ = NULL;
}

// takes CVector and CMat objects. In case of CMat, the samples are ordered as rows
CVisionData* CPCA::exec()
{
	CVisionData working_data = getConcatenatedDataAndClearBuffer();

	if(!opencv_pca_ptr_)
	{
		// TODO: throw NotTrainedException instead!
		cerr << "PCA has to be trained first!" << endl;
		exit(-1);
	}

//	double min, max;
//	minMaxLoc(working_data->data(), &min, &max);
//	cout << "min-max of PCA test data: " << min << " - " << max << endl << flush;

	Mat output;
	opencv_pca_ptr_->project(working_data.data(), output);

	return new CVisionData(output, DATA_TYPE_VECTOR);
}

// train_data contains for each sample one Row and must be already NORMALIZED (value range from 0 to 1)
void CPCA::train()
{
	CVisionData train_data = getConcatenatedDataAndClearBuffer();
	//assert(train_data.data().rows == static_cast<int>(train_labels.data().rows));

	vector<int> non_bg_indices;
	for(int row_cnt = 0; row_cnt < data_labels_->data().rows; ++row_cnt)
		if(data_labels_->data().at<int>(row_cnt, 0))
			non_bg_indices.push_back(row_cnt);

	Mat train_data_non_bg = Mat::zeros(non_bg_indices.size(), train_data.data().cols, train_data.data().type());
	int cur_row = 0;
	for(vector<int>::const_iterator non_bg_it = non_bg_indices.begin(); non_bg_it != non_bg_indices.end(); ++non_bg_it, ++cur_row)
	{
		train_data.data().row(*non_bg_it).copyTo(train_data_non_bg.row(cur_row));
		//cout << "Channels: " << train_data->data().channels() << " Type: " << train_data->signature2str() << endl;
		//Mat temp = train_data->data().row(*non_bg_it).reshape(0,86);
		//cout << "Channels: " << temp.channels() << " Type: " << CVisionData::signature2str(temp.type()) << endl;
		//PAUSE_AND_SHOW(temp);
		//normalize(train_data_non_bg.row(cur_row), train_data_non_bg.row(cur_row), 0.0, 1.0, NORM_MINMAX);
	}

	//normalize(train_data_non_bg, train_data_non_bg, 0.0, 1.0, NORM_MINMAX);
//	double min, max;
//	minMaxLoc(train_data_non_bg, &min, &max);
//	cout << "min-max of PCA training data: " << min << " - " << max << endl << flush;

	cout << "Training PCA... " << flush;
	opencv_pca_ptr_ = new PCA(train_data_non_bg, Mat(), CV_PCA_DATA_AS_ROW, n_eigenvectors_);
	cout << "done!" << endl << flush;

	cout << n_eigenvectors_ << " eigenvectors calculated from " << train_data_non_bg.rows << " samples." << endl;
//	cout << "Projection Matrix size: " << opencv_pca_ptr_->eigenvectors.rows << " x " << opencv_pca_ptr_->eigenvectors.cols << endl;
//	cout << "Matrix snapshot: " << opencv_pca_ptr_->eigenvectors(Rect(0,0,10,10)) << endl;
//
//	cout << "writing eigen images" << endl;
//	for(int v_cnt = 0; v_cnt < opencv_pca_ptr_->eigenvalues.rows; ++v_cnt)
//	{
//		float eigenval = opencv_pca_ptr_->eigenvalues.at<float>(v_cnt);
//		float val_start = -sqrt(eigenval);
//		float val_end = sqrt(eigenval);
//		vector<float> values;
//		linspace<float>(values, val_start, val_end, 1);
//		for(vector<float>::iterator val_it = values.begin(); val_it != values.end(); ++val_it)
//		{
//			Mat eigen_image = Mat(SWIN_SIZE, SWIN_SIZE, CV_32FC1, opencv_pca_ptr_->eigenvectors.row(v_cnt).data) * *val_it;
//			normalize(eigen_image, eigen_image, 0.0, 1.0, NORM_MINMAX);
//			Mat eigen_image_uchar;
//			eigen_image.convertTo(eigen_image_uchar, CV_8UC1, 255);
//			stringstream im_name;
//			im_name << "eigen_image_" << v_cnt << "_" << distance(values.begin(), val_it) << ".jpg";
//			imwrite(im_name.str(), eigen_image_uchar);
//		}
//	}


//	cout << "EigenValues are: " << endl;
//	for(int v_cnt = 0; v_cnt < opencv_pca_ptr_->eigenvalues.rows; ++v_cnt)
//	{
//		cout << opencv_pca_ptr_->eigenvalues.at<float>(v_cnt) << endl;
//	}

	// Computing the average vector (average of all samples per dimension: average of all (0,0) pixels, all (0,1), ...
//	int nsamples = train_data.mat_.rows;
//	int sample_dim = train_data.mat_.cols;

//	Eigen::VectorXf mean_vec(sample_dim);
//	for(int dim_cnt = 0; dim_cnt < sample_dim; ++dim_cnt)
//	{
//		mean_vec(dim_cnt) = static_cast<float>(mean(train_data.mat_.col(dim_cnt)).val[0]);
//	}
//
//	// Subtract the average vector from all samples
//	Eigen::MatrixXf normed_samples(sample_dim, nsamples);
//	const Eigen::Map<const Eigen::MatrixXf> eigen_train_data(reinterpret_cast<const float*>(train_data.mat_.data), sample_dim, nsamples);
//	for(int sample_cnt = 0; sample_cnt < nsamples; ++sample_cnt)
//	{
//		normed_samples.col(sample_cnt) = eigen_train_data.col(sample_cnt) - mean_vec;
////		for(int dim_cnt = 0; dim_cnt < sample_dim; ++dim_cnt)
////		{
////			normed_samples(dim_cnt, sample_cnt) = train_data.mat_.at<float>(dim_cnt, sample_cnt) - mean_vec(dim_cnt);
////		}
//	}
//
//	// get the patternwise (nsamples x nsamples) covariance matrix
//	Eigen::MatrixXf covariance_matrix(nsamples, nsamples);
//	covariance_matrix = normed_samples.transpose() * normed_samples;
//
//	// Get their eigenvectors and eigenvalues
//	Eigen::EigenSolver<Eigen::MatrixXf> es(covariance_matrix, true); // compute eigenvectors inside constructor
//	Eigen::VectorXf eigen_values = es.eigenvalues();
//	Eigen::MatrixXf eigen_vectors = es.eigenvectors();




}

void CPCA::save(FileStorage& fs) const
{
	stringstream config_name_n_eigenvectors;
	config_name_n_eigenvectors << CONFIG_NAME_PCA_N_EIGENVECTORS << "-" << module_id_;
	fs << config_name_n_eigenvectors.str().c_str() << n_eigenvectors_;

	if(opencv_pca_ptr_)
	{
		stringstream config_name_vectors, config_name_values, config_name_means, config_name_norm;
		config_name_vectors << CONFIG_NAME_PCA_EIGENVECTORS << "-" << module_id_;
		config_name_values << CONFIG_NAME_PCA_EIGENVALUES << "-" << module_id_;
		config_name_means << CONFIG_NAME_PCA_MEANS << "-" << module_id_;
		config_name_norm << CONFIG_NAME_PCA_NORM << "-" << module_id_;
		fs << config_name_vectors.str().c_str() << opencv_pca_ptr_->eigenvectors;
		fs << config_name_values.str().c_str()  << opencv_pca_ptr_->eigenvalues;
		fs << config_name_means.str().c_str() << opencv_pca_ptr_->mean;
		//fs << config_name_norm.str().c_str() << norm_;
	}
}

void CPCA::load(FileStorage& fs)
{
	stringstream config_name_n_eigenvectors, config_name_vectors, config_name_values, config_name_means, config_name_norm;
	config_name_n_eigenvectors << CONFIG_NAME_PCA_N_EIGENVECTORS << "-" << module_id_;
	config_name_vectors << CONFIG_NAME_PCA_EIGENVECTORS << "-" << module_id_;
	config_name_values << CONFIG_NAME_PCA_EIGENVALUES << "-" << module_id_;
	config_name_means << CONFIG_NAME_PCA_MEANS << "-" << module_id_;
	config_name_norm << CONFIG_NAME_PCA_NORM << "-" << module_id_;

	fs[config_name_n_eigenvectors.str().c_str()] >> n_eigenvectors_;
	if(opencv_pca_ptr_)
		delete opencv_pca_ptr_;

	opencv_pca_ptr_ = new PCA();
	fs[config_name_vectors.str().c_str()] >> opencv_pca_ptr_->eigenvectors;
	fs[config_name_values.str().c_str()] >> opencv_pca_ptr_->eigenvalues;
	fs[config_name_means.str().c_str()] >> opencv_pca_ptr_->mean;
	//fs[config_name_norm.str().c_str()] >> norm_;
}
