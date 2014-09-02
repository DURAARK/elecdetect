#include "BinaryContours.h"

CBinaryContours::CBinaryContours(MODULE_CONSTRUCTOR_SIGNATURE)
{
	module_print_name_ = "Canny";
	required_input_signature_ = DATA_TYPE_IMAGE | CV_8UC1; // takes single channel grayscale image as input only
	output_signature_ = DATA_TYPE_IMAGE | CV_8UC1;

	if(is_root)
		setAsRoot();
}


CBinaryContours::~CBinaryContours()
{
}

CVisionData* CBinaryContours::exec()
{
	CVisionData working_data = getConcatenatedDataAndClearBuffer();

	Mat working_image;
	Canny(working_data.data(), working_image, 20.0, 160.0);
	return new CVisionData(working_image, DATA_TYPE_IMAGE);

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

	
}

void CBinaryContours::save(FileStorage& fs) const
{

}
void CBinaryContours::load(FileStorage& fs)
{

}
