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

#include "VisionModule.h"

#define CONFIG_NAME_RF  "RandomForest"

using namespace std;
using namespace cv;

class CRandomForest: public CVisionModule
{
private:
	CvRTrees* rf_;
	CvRTParams* rf_params_;
	CRandomForest();

public:
	CRandomForest(MODULE_CONSTRUCTOR_SIGNATURE);
	virtual ~CRandomForest();

	CVisionData* exec();
	void train();
	void save(FileStorage& fs) const;
	void load(FileStorage& fs);
};

#endif /* RANDOMFOREST_H_ */
