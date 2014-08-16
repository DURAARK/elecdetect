#pragma once

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>


#define DATA_TYPE_SCALAR        1 << 16
#define DATA_TYPE_VECTOR        1 << 17
#define DATA_TYPE_IMAGE         1 << 18
#define DATA_TYPE_POINT         1 << 19

#define DATA_ARRAY              1 << 24

#define DATA_FORMAT_MASK        0x00ff
#define DATA_TYPE_MASK          0x0f00

#define DISPLAY_WIN_NAME        "VisionData"

using namespace std;
using namespace cv;

class CVisionData
{
private:
	int data_signature_; // data signature is a combination of data type(s) and format (and has a flag if its an array)
	CVisionData();
	CVisionData(const CVisionData& other);

	Mat mat_;

public:
	CVisionData(const Mat& data, const int& type) : mat_(data)
    {
		data_signature_ = type | mat_.type();
    };

	~CVisionData() { };

//	inline CVisionData& clone()
//	{
//		return CVisionData(mat_.clone(), data_signature_ & DATA_TYPE_MASK);
//	}

	void assignData(const Mat& data, const int& type)
	{
		mat_ = data;
		data_signature_ = type | mat_.type();
		if((data_signature_ & DATA_TYPE_VECTOR) && (mat_.rows > 1))
			data_signature_ |= DATA_ARRAY;
	}

//	Mat& data()
//	{
//		return mat_;
//	}

	const Mat& data() const
	{
		return mat_;
	}

	inline void show() const
	{
		switch(data_signature_ & DATA_TYPE_MASK)
		{
		case DATA_TYPE_IMAGE:
			Mat temp;
			if((mat_.type() & CV_MAT_DEPTH_MASK) == CV_32F)
				mat_.convertTo(temp, CV_8U, 255);
			else
				temp = mat_;
			std::stringstream win_name;
			win_name << DISPLAY_WIN_NAME;
			cv::imshow(win_name.str(), temp);
			break;
		}

		cout << "Data Type is: " << type2str();

	}

	inline string type2str() const
	{
		string r;

		uchar depth = mat_.type() & CV_MAT_DEPTH_MASK;
		uchar chans = 1 + (mat_.type() >> CV_CN_SHIFT);

		switch ( depth ) {
		case CV_8U:  r = "8U"; break;
		case CV_8S:  r = "8S"; break;
		case CV_16U: r = "16U"; break;
		case CV_16S: r = "16S"; break;
		case CV_32S: r = "32S"; break;
		case CV_32F: r = "32F"; break;
		case CV_64F: r = "64F"; break;
		default:     r = "User"; break;
		}

		r += "C";
		r += (chans+'0');
		r+= ":";

		switch ( data_signature_ & DATA_TYPE_MASK ) {
		case DATA_TYPE_SCALAR:       r += "Scalar"; break;
		case DATA_TYPE_VECTOR:       r += "Vector"; break;
		case DATA_TYPE_IMAGE:        r += "Image"; break;
		case DATA_TYPE_POINT:        r += "Point"; break;
		default:                     r += "Unknown"; break;
		}

		if( data_signature_ | DATA_ARRAY )
			r += "Array";

		return r;
	}

	inline int getType() const // return f.i. DATA_TYPE_IMAGE
	{
		return data_signature_ & DATA_TYPE_MASK;
	}

	inline int getFormat() const // returns f.i. CV_32FC1
	{
		return data_signature_ & DATA_FORMAT_MASK;
	}

	inline bool isArray() const
	{
		return data_signature_ & DATA_ARRAY;
	}

	inline int getSignature() const // but without the Array-Flag
	{
		return data_signature_ & (DATA_TYPE_MASK | DATA_FORMAT_MASK);
	}
};


// DATA CONVERTER Class for automatic data conversion before Vision Modules execute their task

class CDataConverter
{
private:
	CDataConverter();
	vector<void(*)(CVisionData& data)> convert_functions_;
	int expected_input_signature_;

	void convImage2Vector(CVisionData& data)
	{
		data.assignData(data.data().reshape(0,1), DATA_TYPE_VECTOR);
	}
	void convRGB2Gray(CVisionData& data)
	{
		Mat temp;
		cvtColor(data.data(), temp, CV_BGR2GRAY, 1);
		data.assignData(temp, data.getType());
	}
	void convUChar2Float(CVisionData& data)
	{
		Mat temp;
		data.data().convertTo(temp, CV_32F, 1.0/255.0);
		data.assignData(temp, data.getType());
	}

public:
	CDataConverter(const int& input_data_signature, const int& output_data_signature) : expected_input_signature_(input_data_signature)
	{
		// TYPE conversions
		int input_data_type = (input_data_signature & DATA_TYPE_MASK);
		int output_data_type = (output_data_signature & DATA_TYPE_MASK);
		if(input_data_type != output_data_type)
		{
			// Possible Type conversions
			// -------------------------

			// convert image to vector by concatenating all pixels
			if(input_data_type == DATA_TYPE_IMAGE && output_data_type == DATA_TYPE_VECTOR)
			{
				convert_functions_.push_back(&convImage2Vector);
			}
		}

		// FORMAT conversions
		int input_data_format = (input_data_signature & DATA_FORMAT_MASK);
		int output_data_format = (output_data_signature & DATA_FORMAT_MASK);
		if(input_data_format != output_data_format)
		{
			// Possible format conversions
			// ---------------------------

			// RGB values to Grayscale (3 channel -> 1 channel)
			if(((1 + (input_data_format >> CV_CN_SHIFT)) == 3) && ((1 + (input_data_format >> CV_CN_SHIFT)) == 1))
			{
				convert_functions_.push_back(&convRGB2Gray);
			}

			// 8bit to floating point (with range scaling from [0..255] -> [0..1])
			if((input_data_format & CV_MAT_DEPTH_MASK) == CV_8U && (output_data_format & CV_MAT_DEPTH_MASK) == CV_32F)
			{
				convert_functions_.push_back(&convUChar2Float);
			}
		}

		if(convert_functions_.empty())
		{
			cout << "No suitable conversion found! Throwing PipeConfigException" << endl;
			exit(-1);
		}
	};

	~CDataConverter() { };

	void convert(CVisionData& data)
	{
		if(data.getSignature() != expected_input_signature_)
		{
			cout << "Are you kidding me?!" << endl;
			exit(-1);
		}
		vector<void(*)(CVisionData& data)>::const_iterator convert_routines_it;
		for(convert_routines_it = convert_functions_.begin(); convert_routines_it != convert_functions_.end(); ++convert_routines_it)
		{
			(*convert_routines_it)(data);
		}
	}

};

