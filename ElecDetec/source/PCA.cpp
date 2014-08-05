/*
 * PCA.cpp
 *
 *  Created on: Aug 4, 2014
 *      Author: test
 */

#include "PCA.h"

CPCA::CPCA() : n_eigenvectors_(DEFAULT_NUMBER_OF_EIGENVECTORS)
{

}

CPCA::~CPCA()
{

}

void CPCA::exec(std::vector<CVisionData*>& data) throw(VisionDataTypeException)
{
	if(data.back()->getType() != TYPE_VECTOR)
		throw(VisionDataTypeException(data.back()->getType(), TYPE_VECTOR));
}

// train_data contains for each sample one Row
void CPCA::train(const CMat& train_data, const CVector<int>& train_labels) throw(VisionDataSizeException)
{
	if(train_data.mat_.rows != static_cast<int>(train_labels.vec_.size()))
		throw(VisionDataSizeException(train_labels.vec_.size(), train_data.mat_.rows));

	assert(train_data.mat_.type() == CV_32FC1);

	// Computing the average vector (average of all samples per dimension: average of all (0,0) pixels, all (0,1), ...
	int nsamples = train_data.mat_.rows;
	int sample_dim = train_data.mat_.cols;
	Eigen::VectorXf mean_vec(sample_dim);
	for(int dim_cnt = 0; dim_cnt < sample_dim; ++dim_cnt)
	{
		mean_vec(dim_cnt) = static_cast<float>(mean(train_data.mat_.col(dim_cnt)).val[0]);
	}



}

void CPCA::save(FileStorage& fs) const
{
	fs << CONFIG_NAME_PCA_N_EIGENVECTORS << n_eigenvectors_;
}

void CPCA::load(FileStorage& fs)
{
	fs[CONFIG_NAME_PCA_N_EIGENVECTORS] >> n_eigenvectors_;
}
