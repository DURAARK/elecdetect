/*
 * ColorChannel.h
 *
 *  Created on: Oct 8, 2014
 *      Author: test
 */

#ifndef COLORCHANNEL_H_
#define COLORCHANNEL_H_

#include "VisionModule.h"
#include "Defines.h"
#include "Utils.h"

using namespace std;
using namespace cv;

class CColorChannel: public CVisionModule
{
private:
	CColorChannel();

	int channel_nr_; // channel number which should be extracted
public:
	CColorChannel(MODULE_CONSTRUCTOR_SIGNATURE);
	virtual ~CColorChannel();

	virtual CVisionData* exec();
	virtual void save(FileStorage& fs) const;
	virtual void load(FileStorage& fs);
};

#endif /* COLORCHANNEL_H_ */
