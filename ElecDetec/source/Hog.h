/*
 * ElecDetec: Hog.h
 *
 *  Created on: Feb, 2015
 *      Author: Robert Viehauser
 */


#ifndef HOG_H_
#define HOG_H_

#include <opencv2/opencv.hpp>
#include <Defines.h>

#define WIN_SIZE         128  // Hog window size
#define CELL_SIZE          8  // Hog Cell Size: window size must be divisible by the cell Size
#define BLOCK_SIZE        16  // Hog Block Size: must be a multiple of cell size
#define BLOCK_STRIDE       8  // Hog Block Stride: window size must be divisible by block stride
#define NBINS              9  // Number of bins of the histograms

using namespace std;
using namespace cv;

class CHog
{
private:
    HOGDescriptor* hogy_;

    Size win_size_;
    Size cell_size_;
    Size block_size_;
    Size block_stride_;
	int nbins_;

    void visualize(cv::Mat& img, std::vector<float>& descriptorValues);

public:
    CHog();
	~CHog();

    void extractFeatureVector(const Mat& input_img, vector<float>& output_vec);
};

#endif
