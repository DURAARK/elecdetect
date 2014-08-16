///*
// * DummyFeature.h
// *
// *  Created on: Jul 4, 2014
// *      Author: test
// *
// *      This feature module rescales the image patch and converts the pixels directly
// *      to a feature vector
// */
//
//#ifndef DUMMYFEATURE_H_
//#define DUMMYFEATURE_H_
//
//#include "FeatureExtractorModule.h"
//#include "Vector.h"
//#include "Mat.h"
//
//using namespace std;
//using namespace cv;
//
//class CDummyFeature: public CFeatureExtractorModule
//{
//private:
//	// defines the feature length and thus the scale of the resize process
//	Size canonical_size_;
//	int feature_length_;
//
//public:
//	CDummyFeature();
//	virtual ~CDummyFeature();
//
//	virtual void exec(std::vector<CVisionData*>& data) throw(VisionDataTypeException);
//	virtual void save(FileStorage& fs) const;
//	virtual void load(FileStorage& fs);
//	virtual int getFeatureLength();
//};
//
//#endif /* DUMMYFEATURE_H_ */
