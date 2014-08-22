/*
 * DistanceTransform.h
 *
 *  Created on: Jun 30, 2014
 *      Author: test
 */

#ifndef DISTANCETRANSFORM_H_
#define DISTANCETRANSFORM_H_

#include "VisionModule.h"
#include "VisionData.h"

using namespace std;
using namespace cv;

class CDistanceTransform: public CVisionModule
{
private:
	CDistanceTransform();

public:
	CDistanceTransform(MODULE_CONSTRUCTOR_SIGNATURE);
	virtual ~CDistanceTransform();

	virtual CVisionData* exec();
	virtual void save(FileStorage& fs) const;
	virtual void load(FileStorage& fs);
};

#endif /* DISTANCETRANSFORM_H_ */
