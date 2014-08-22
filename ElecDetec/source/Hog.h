#ifndef HOG_H_
#define HOG_H_


#include "VisionModule.h"
#include "Mat.h"
#include "Vector.h"

#include <opencv2/opencv.hpp>

class CHog : public CVisionModule
{
private:
	CHog();

	cv::HOGDescriptor* hogy_;

	cv::Size win_size_;
	cv::Size cell_size_;
	cv::Size block_size_;
	cv::Size block_stride_;
	int nbins_;

public:
	CHog(MODULE_CONSTRUCTOR_SIGNATURE);
	~CHog();

	CVisionData* exec();
	void save(FileStorage& fs) const;
	void load(FileStorage& fs);
//	virtual int getFeatureLength();

	void visualize(cv::Mat& img, std::vector<float>& descriptorValues);
};

#endif
