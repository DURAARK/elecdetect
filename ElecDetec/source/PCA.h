/*
 * PCA.h
 *
 *  Created on: Aug 4, 2014
 *      Author: test
 */

#ifndef PCA_H_
#define PCA_H_


#include "Vector.h"
#include "Mat.h"

#include "SubspaceModule.h"

#define DEFAULT_NUMBER_OF_EIGENVECTORS  90

#define CONFIG_NAME_PCA_N_EIGENVECTORS   "number-of-pca-eigenvectors"
#define CONFIG_NAME_PCA_EIGENVALUES      "opencv-pca-eigenvalues"
#define CONFIG_NAME_PCA_EIGENVECTORS     "opencv-pca-eigenvectors"
#define CONFIG_NAME_PCA_MEANS            "opencv-pca-means"

class CPCA: public CSubspaceModule
{
private:
	int n_eigenvectors_;
	PCA* opencv_pca_ptr_;

public:
	CPCA();
	virtual ~CPCA();

	void exec(std::vector<CVisionData*>& data) throw(VisionDataTypeException);
	void train(const CMat& train_data, const CVector<int>& train_labels);
	void save(FileStorage& fs) const;
	void load(FileStorage& fs);
};

#endif /* PCA_H_ */
