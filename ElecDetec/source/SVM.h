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

#define CONFIG_NAME_SVM  "SVM"

using namespace std;
using namespace cv;

class CSVM: public CVisionModule
{
private:
	CvSVM* svm_;
	CvSVMParams* svm_params_;
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
