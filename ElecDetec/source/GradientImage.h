/*
 * GradientImage.h
 *
 *  Created on: Jun 30, 2014
 *      Author: test
 */

#ifndef GRADIENTIMAGE_H_
#define GRADIENTIMAGE_H_

#include "VisionModule.h"
#include "Mat.h"

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
	CGradientImage(int inchain_input_signature);
	virtual ~CGradientImage();

	virtual void exec(const CVisionData& input_data, CVisionData& output_data);
	virtual void save(FileStorage& fs) const;
	virtual void load(FileStorage& fs);
};

#endif /* GRADIENTIMAGE_H_ */
