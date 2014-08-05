/*
 * PCA.h
 *
 *  Created on: Aug 4, 2014
 *      Author: test
 */

#ifndef PCA_H_
#define PCA_H_

#include "Eigen/Eigen"

#include "Vector.h"
#include "Mat.h"

#include "SubspaceModule.h"

#define DEFAULT_NUMBER_OF_EIGENVECTORS  100

#define CONFIG_NAME_PCA_N_EIGENVECTORS   "number-of-pca-eigenvectors"

class CPCA: public CSubspaceModule
{
private:
	int n_eigenvectors_;
	Eigen::MatrixXf w_pca_;


public:
	CPCA();

	virtual ~CPCA();

	void exec(std::vector<CVisionData*>& data) throw(VisionDataTypeException);

	// train_data contains for each sample one CVisionData and train_labels a CVisionData-Label
	void train(const CMat& train_data, const CVector<int>& train_labels) throw(VisionDataSizeException);

	void save(FileStorage& fs) const;

	void load(FileStorage& fs);
};

#endif /* PCA_H_ */
