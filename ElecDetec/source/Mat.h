//#ifndef MAT_H_
//#define MAT_H_
//
//
//#include "VisionData.h"
//
//#include <opencv2/opencv.hpp>
//
//#define SHOW_WIN_NAME "CMat"
//
//using namespace std;
//using namespace cv;
//
//class CMat :
//	public CVisionData
//{
//private:
//	//static int show_instance_cnt_;
//
//public:
//	inline CMat() { this->type_ = TYPE_MAT; };
//	inline CMat(Mat& cv_mat)
//	{
//		this->type_ = TYPE_MAT;  mat_ = cv_mat;
//	};
//
//	inline ~CMat() { };
//
//	cv::Mat mat_;
//
//	inline virtual void show()
//	{
//		std::stringstream win_name;
//		win_name << SHOW_WIN_NAME;
//		cv::imshow(win_name.str(), mat_);
//	}
//
//	inline string type2str() const
//	{
//		string r;
//
//		uchar depth = mat_.type() & CV_MAT_DEPTH_MASK;
//		uchar chans = 1 + (mat_.type() >> CV_CN_SHIFT);
//
//		switch ( depth ) {
//		case CV_8U:  r = "8U"; break;
//		case CV_8S:  r = "8S"; break;
//		case CV_16U: r = "16U"; break;
//		case CV_16S: r = "16S"; break;
//		case CV_32S: r = "32S"; break;
//		case CV_32F: r = "32F"; break;
//		case CV_64F: r = "64F"; break;
//		default:     r = "User"; break;
//		}
//
//		r += "C";
//		r += (chans+'0');
//
//		return r;
//	}
//};
//
//#endif

