/*
 * Brief.h
 *
 *  Created on: Jul 7, 2014
 *      Author: test
 */

#ifndef OWNBRIEF_H_
#define OWNBRIEF_H_

#include <time.h>
#include <opencv2/opencv.hpp>

#include "Defines.h"
#include "VisionModule.h"
#include "VisionData.h"
#include "Exceptions.h"

using namespace std;
using namespace cv;

typedef pair<Point2f, Point2f> test_pair;

class COwnBrief: public CVisionModule
{
private:
	int feature_length_;
	vector<test_pair> rel_test_pairs_;

	COwnBrief();

	float compare(const Mat& img0, const Point2i& pt1, const Point2i& pt2) const;
	void initTestPairs();

public:
	COwnBrief(MODULE_CONSTRUCTOR_SIGNATURE);
	virtual ~COwnBrief();

	virtual CVisionData* exec();
	virtual void save(FileStorage& fs) const;
	virtual void load(FileStorage& fs);

};

#endif /* OWNBRIEF_H_ */
