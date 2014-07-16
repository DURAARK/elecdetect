/*
 * GradientImage.h
 *
 *  Created on: Jun 30, 2014
 *      Author: test
 */

#ifndef GRADIENTIMAGE_H_
#define GRADIENTIMAGE_H_

#include "PreprocessingModule.h"
#include "Mat.h"

using namespace std;
using namespace cv;

class CGradientImage: public CPreprocessingModule
{
private:
	int scale_;
	int delta_;
	int ddepth_;

public:
	CGradientImage();
	virtual ~CGradientImage();

	virtual void exec(std::vector<CVisionData*>& data) throw(VisionDataTypeException);
	virtual void save(FileStorage& fs) const;
	virtual void load(FileStorage& fs);
};

#endif /* GRADIENTIMAGE_H_ */
