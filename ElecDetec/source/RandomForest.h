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

class CRandomForest: public CVisionModule
{
private:
	CvRTrees* rf_;
	CvRTParams* rf_params_;
	CRandomForest();

public:
	CRandomForest(int inchain_input_signature);
	virtual ~CRandomForest();

	void exec(const CVisionData& input_data, CVisionData& output_data);
	void train(const CVisionData& train_data, const CVisionData& train_labels);
	void save(FileStorage& fs) const;
	void load(FileStorage& fs);
};

#endif /* RANDOMFOREST_H_ */
