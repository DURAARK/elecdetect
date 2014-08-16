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

#include "VisionModule.h"

#define DEFAULT_NUMBER_OF_EIGENVECTORS  90

#define CONFIG_NAME_PCA_N_EIGENVECTORS   "number-of-pca-eigenvectors"
#define CONFIG_NAME_PCA_EIGENVALUES      "opencv-pca-eigenvalues"
#define CONFIG_NAME_PCA_EIGENVECTORS     "opencv-pca-eigenvectors"
#define CONFIG_NAME_PCA_MEANS            "opencv-pca-means"

class CPCA: public CVisionModule
{
private:
	int n_eigenvectors_;
	PCA* opencv_pca_ptr_;

	CPCA();

public:
	CPCA(int inchain_input_signature);
	virtual ~CPCA();

	void exec(const CVisionData& input_data, CVisionData& output_data);
	void train(const CVisionData& train_data, const CVisionData& train_labels);
	void save(FileStorage& fs) const;
	void load(FileStorage& fs);
};

#endif /* PCA_H_ */
