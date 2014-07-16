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

#include "FeatureExtractorModule.h"
#include "VisionData.h"
#include "Exceptions.h"
#include "Mat.h"
#include "Vector.h"

#define DEFAULT_FEATURE_LENGTH 1024

#define CONFIG_NAME_TESTPAIRS         "brief-testpairs"

using namespace std;
using namespace cv;

typedef pair<Point2f, Point2f> test_pair;

class COwnBrief: public CFeatureExtractorModule
{
private:
	int feature_length_;
	vector<test_pair> rel_test_pairs_;

	float compare(const Mat& img0, const Point2i& pt1, const Point2i& pt2) const;
	void initTestPairs();

public:
	COwnBrief();
	virtual ~COwnBrief();

	virtual void exec(std::vector<CVisionData*>& data) throw(VisionDataTypeException);
	virtual void save(FileStorage& fs) const;
	virtual void load(FileStorage& fs);

	inline int getFeatureLength()
	{
		return feature_length_;
	}
};

#endif /* OWNBRIEF_H_ */
