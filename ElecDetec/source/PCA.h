/*
 * PCA.h
 *
 *  Created on: Aug 4, 2014
 *      Author: test
 */

#ifndef PCA_H_
#define PCA_H_

#include "Debug.h"
#include "VisionData.h"
#include "VisionModule.h"
#include "Utils.h"

#define DEFAULT_NUMBER_OF_EIGENVECTORS  70

#define CONFIG_NAME_PCA_N_EIGENVECTORS   "number-of-pca-eigenvectors"
#define CONFIG_NAME_PCA_EIGENVALUES      "opencv-pca-eigenvalues"
#define CONFIG_NAME_PCA_EIGENVECTORS     "opencv-pca-eigenvectors"
#define CONFIG_NAME_PCA_MEANS            "opencv-pca-means"
#define CONFIG_NAME_PCA_NORM             "opencv-pca-norm"

class CPCA: public CVisionModule
{
private:
	int n_eigenvectors_;
	PCA* opencv_pca_ptr_;

	CPCA();

public:
	CPCA(MODULE_CONSTRUCTOR_SIGNATURE);
	virtual ~CPCA();

	CVisionData* exec();
	void train();
	void save(FileStorage& fs) const;
	void load(FileStorage& fs);
};

#endif /* PCA_H_ */
