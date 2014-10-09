/*
 * QGradient.h
 *
 *  Created on: Okt 6, 2014
 *      Author: test
 */

#ifndef QGRADIENT_H_
#define QGRADIENT_H_

#include <math.h>

#include "Debug.h"
#include "VisionModule.h"

using namespace std;
using namespace cv;

class CQuantizedGradient: public CVisionModule
{
private:
	CQuantizedGradient();

	// Gradient Quantization Paramters
	int n_bins_;
	float bin_step_;
	int n_neigbours_;
	float mag_threshold_;

	// Sobel parameters
	int scale_;
	int delta_;
	int ddepth_;

public:
	CQuantizedGradient(MODULE_CONSTRUCTOR_SIGNATURE);
	virtual ~CQuantizedGradient();

	virtual CVisionData* exec();
	virtual void save(FileStorage& fs) const;
	virtual void load(FileStorage& fs);
};

#endif /* QGRADIENT_H_ */
