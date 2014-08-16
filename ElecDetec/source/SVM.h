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
	CSVM(int inchain_input_signature);
	virtual ~CSVM();

	void exec(const CVisionData& input_data, CVisionData& output_data);
	void train(const CVisionData& train_data, const CVisionData& train_labels);
	void save(FileStorage& fs) const;
	void load(FileStorage& fs);
};

#endif /* SVM_H_ */
