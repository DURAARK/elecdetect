/*
 * DistanceTransform.h
 *
 *  Created on: Jun 30, 2014
 *      Author: test
 */

#ifndef DISTANCETRANSFORM_H_
#define DISTANCETRANSFORM_H_

#include "PreprocessingModule.h"
#include "Mat.h"

using namespace std;
using namespace cv;

class CDistanceTransform: public CPreprocessingModule
{
public:
	CDistanceTransform();
	virtual ~CDistanceTransform();

	virtual void exec(std::vector<CVisionData*>& data) throw(VisionDataTypeException);
	virtual void save(FileStorage& fs) const;
	virtual void load(FileStorage& fs);
};

#endif /* DISTANCETRANSFORM_H_ */
