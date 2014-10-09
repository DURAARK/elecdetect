/*
 * ColorChannel.cpp
 *
 *  Created on: Oct 8, 2014
 *      Author: test
 */

#include "ColorChannel.h"

CColorChannel::CColorChannel(MODULE_CONSTRUCTOR_SIGNATURE) :
channel_nr_(0)
{
	// takes color image as input and produces also one single channel image
	MODULE_CTOR_INIT("Orientation Filter", DATA_TYPE_IMAGE | CV_8UC3, DATA_TYPE_IMAGE | CV_8UC1);

	string channel_str = COLOR_CHANNEL_DEFAULT_CHANNEL;
	if(!module_params.empty())
	{
		vector<string> params_vec = splitStringByDelimiter(module_params, MODULE_PARAM_DELIMITER);
		if(params_vec.size() == 1)
		{
			channel_str = params_vec[0];
		}
		else
		{
			cerr << "Only one color channel filter param allowed!";
			exit(-1);
		}
	}

	char c_channel = toupper(*channel_str.begin());
	switch(c_channel)
	{
	case 'R':
		channel_nr_ = 2;
		break;
	case 'G':
		channel_nr_ = 1;
		break;
	case 'B':
		channel_nr_ = 0;
		break;
	default:
		cerr << "Color channel parameter not recognized. Using default." << endl;
		exit(-1);
	}

}

CColorChannel::~CColorChannel()
{

}

CVisionData* CColorChannel::exec()
{
	CVisionData working_data = getConcatenatedDataAndClearBuffer();
	Mat* working_data_ptr = new Mat();
	*working_data_ptr = working_data.data();

	//normalize(out_img->mat_, out_img->mat_, 0, 255, NORM_MINMAX);
	Mat out_img = Mat::zeros(working_data.data().size(), CV_8UC1);
	Mat* out_img_ptr = &out_img;
	int from_to[] = { channel_nr_,0 };
	mixChannels(working_data_ptr, 1, out_img_ptr, 1, from_to, 1);
//	imshow("orig", working_data.data());
//	PAUSE_AND_SHOW(out_img);
	delete working_data_ptr;
	return new CVisionData(out_img, DATA_TYPE_IMAGE);
}

void CColorChannel::save(FileStorage& fs) const
{
	stringstream config_name;
	config_name << COLOR_CHANNEL_CONFIG_NAME << "-" << module_id_;
	fs << config_name.str().c_str() << "{";
	fs << "channel-nr" << channel_nr_;
	fs << "}";
}

void CColorChannel::load(FileStorage& fs)
{
	stringstream config_name;
	config_name << COLOR_CHANNEL_CONFIG_NAME << "-" << module_id_;
	FileNode fn = fs[config_name.str().c_str()];
	fn["channel-nr"] >> channel_nr_;
}
