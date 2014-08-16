#include "BinaryContours.h"

CBinaryContours::CBinaryContours(int inchain_input_signature)
{
	module_print_name_ = "Canny";
	required_input_signature_mask_ = DATA_TYPE_IMAGE | CV_8UC1; // takes single channel grayscale image as input only
	output_type_ = DATA_TYPE_IMAGE | CV_8UC1;

	if(inchain_input_signature != required_input_signature_mask_)
	{
		data_converter_ = new CDataConverter(inchain_input_signature, required_input_signature_mask_);
	}
}


CBinaryContours::~CBinaryContours()
{
}

void CBinaryContours::exec(const CVisionData& input_data, CVisionData& output_data)
{
	output_data.assignData(input_data.data(), input_data.getType());
	if(data_converter_)
	{
		data_converter_->convert(output_data);
	}

	assert(output_data.getSignature == required_input_signature_mask_);

	Mat working_image;
	cv::Canny(output_data.data(), working_image, 20.0, 160.0);
	output_data.assignData(working_image, DATA_TYPE_IMAGE);

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
