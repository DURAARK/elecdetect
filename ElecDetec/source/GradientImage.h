/*
 * GradientImage.h
 *
 *  Created on: Jun 30, 2014
 *      Author: test
 */

#ifndef GRADIENTIMAGE_H_
#define GRADIENTIMAGE_H_

#include "Debug.h"
#include "VisionModule.h"

using namespace std;
using namespace cv;

class CGradientImage: public CVisionModule
{
private:
	CGradientImage();

	int scale_;
	int delta_;
	int ddepth_;

public:
	CGradientImage(MODULE_CONSTRUCTOR_SIGNATURE);
	virtual ~CGradientImage();

	virtual CVisionData* exec();
	virtual void save(FileStorage& fs) const;
	virtual void load(FileStorage& fs);
};

#endif /* GRADIENTIMAGE_H_ */
