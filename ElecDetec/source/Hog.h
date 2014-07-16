#ifndef HOG_H_
#define HOG_H_


#include "FeatureExtractorModule.h"
#include "Mat.h"
#include "Vector.h"

#include <opencv2/opencv.hpp>

class CHog :
	public CFeatureExtractorModule
{
private:
	cv::HOGDescriptor* hogy_;

	cv::Size win_size_;
	cv::Size cell_size_;
	cv::Size block_size_;
	cv::Size block_stride_;
	int nbins_;

public:
	CHog();
	~CHog();

	virtual void exec(std::vector<CVisionData*>& data) throw(VisionDataTypeException);
	virtual void save(FileStorage& fs) const;
	virtual void load(FileStorage& fs);
	virtual int getFeatureLength();

	void visualize(cv::Mat& img, std::vector<float>& descriptorValues);
};

#endif
