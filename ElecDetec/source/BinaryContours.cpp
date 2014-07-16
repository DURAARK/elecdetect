#include "BinaryContours.h"

CBinaryContours::CBinaryContours()
{
	module_name_ = "Canny";
}


CBinaryContours::~CBinaryContours()
{
}

void CBinaryContours::exec(std::vector<CVisionData*>& data) throw(VisionDataTypeException)
{
	CMat* img0_ptr = (CMat*)data.back();
	CMat* out_img = new CMat();



	cv::Canny(img0_ptr->mat_, out_img->mat_, 20.0, 160.0);

//	Mat rot;
//	// 90 deg
//	transpose(out_img->mat_, rot);
//	flip(rot, rot, 0);
//	out_img->mat_ = out_img->mat_ + rot;
//
//	// 180 deg
//	transpose(rot, rot);
//	flip(rot, rot, 0);
//	out_img->mat_ = out_img->mat_ + rot;
//
//	// 270 deg
//	transpose(out_img->mat_, rot);
//	flip(rot, rot, 0);
//	out_img->mat_ = out_img->mat_ + rot;


//	cv::distanceTransform(255-out_img->mat_, out_img->mat_, CV_DIST_L2, 3);
//	out_img->mat_.convertTo(out_img->mat_, CV_8UC1);
//	out_img->mat_ = 5*out_img->mat_;
//	GaussianBlur(out_img->mat_, out_img->mat_, Size(21,21), 5.4, 5.4, BORDER_REFLECT);


	data.push_back(out_img);

	
}

void CBinaryContours::save(FileStorage& fs) const
{

}
void CBinaryContours::load(FileStorage& fs)
{

}
