/*
 * Lin_SVM.h
 *
 *  Created on: Sep 29, 2014
 *      Author: test
 */

#ifndef LIN_SVM_H_
#define LIN_SVM_H_

#include "VisionModule.h"
#include "VisionData.h"
#include "Scalar.h"
#include "Vector.h"
#include "VectorArray.h"

#include "libsvm-dense/svm.h"

#define CONFIG_NAME_LIN_SVM  "LinSVM"

using namespace std;
using namespace cv;

class CLinSVM: public CVisionModule
{
private:
	svm_parameter* svm_params_;
	svm_problem* svm_problem_;
	svm_model* svm_model_;

	static const char* svm_type_table_[];
	static const char* kernel_type_table_[];

	CLinSVM();

public:
	CLinSVM(MODULE_CONSTRUCTOR_SIGNATURE);
	virtual ~CLinSVM();

	CVisionData* exec();
	void train();
	void save(FileStorage& fs) const;
	void load(FileStorage& fs);
};

#endif /* LIN_SVM_H_ */
