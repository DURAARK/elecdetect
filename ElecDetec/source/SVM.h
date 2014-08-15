/*
 * SVM.h
 *
 *  Created on: Jun 17, 2014
 *      Author: test
 */

#ifndef SVM_H_
#define SVM_H_

#include "VisionModule.h"
#include "ClassifierModule.h"
#include "VisionData.h"
#include "Scalar.h"
#include "Vector.h"
#include "VectorArray.h"

#define CONFIG_NAME_SVM  "SVM"

using namespace std;
using namespace cv;

class CSVM: public CClassifierModule
{
private:
	CvSVM* svm_;
	CvSVMParams* svm_params_;

public:
	CSVM();
	virtual ~CSVM();

	void exec(vector<CVisionData*>& data) throw(VisionDataTypeException);
	void train(const CMat& train_data, const CVector<int>& train_labels);
	void save(FileStorage& fs) const;
	void load(FileStorage& fs);
};

#endif /* SVM_H_ */
