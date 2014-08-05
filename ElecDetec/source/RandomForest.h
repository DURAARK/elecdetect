/*
 * RandomForest.h
 *
 *  Created on: Jul 8, 2014
 *      Author: test
 */

#ifndef RANDOMFOREST_H_
#define RANDOMFOREST_H_

#include <opencv2/opencv.hpp>
#include <time.h>

#include "ClassifierModule.h"

#define CONFIG_NAME_RF  "RandomForest"

using namespace std;
using namespace cv;

class CRandomForest: public CClassifierModule
{
private:
	CvRTrees* rf_;
	CvRTParams* rf_params_;

public:
	CRandomForest();
	virtual ~CRandomForest();

	virtual void exec(std::vector<CVisionData*>& data) throw(VisionDataTypeException);

	// train_data contains for each sample one CVisionData and train_labels a CVisionData-Label
	virtual void train(const CMat& train_data, const CVector<int>& train_labels) throw(VisionDataSizeException);

	virtual void save(FileStorage& fs) const;

	virtual void load(FileStorage& fs);
};

#endif /* RANDOMFOREST_H_ */
