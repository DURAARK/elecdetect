/*
 * SVM.h
 *
 *  Created on: Jun 17, 2014
 *      Author: test
 */

#ifndef SVM_H_
#define SVM_H_

#include "VisionModule.h"
#include "VisionData.h"
#include "Scalar.h"
#include "Vector.h"
#include "VectorArray.h"
#include "Utils.h"
#include "Debug.h"

#include "libsvm-dense/svm.h"

#define CONFIG_NAME_SVM  "SVM"

using namespace std;
using namespace cv;

class CSVM: public CVisionModule
{
private:
	svm_parameter* svm_params_;
	svm_problem* svm_problem_;
	svm_model* svm_model_;

	bool auto_train_;

	static const char* svm_type_table_[];
	static const char* kernel_type_table_[];

	CSVM();

public:
	CSVM(MODULE_CONSTRUCTOR_SIGNATURE);
	virtual ~CSVM();

	CVisionData* exec();
	void train();
	void save(FileStorage& fs) const;
	void load(FileStorage& fs);
};

#endif /* SVM_H_ */
