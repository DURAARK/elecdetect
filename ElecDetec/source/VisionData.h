#pragma once

#include <iostream>
#include <string>
#include <bitset>
#include <opencv2/opencv.hpp>

#include "Debug.h"

#define DATA_TYPE_SCALAR                 1 << 16
#define DATA_TYPE_WEIGHTED_SCALAR        1 << 17
#define DATA_TYPE_VECTOR                 1 << 18
#define DATA_TYPE_IMAGE                  1 << 19
#define DATA_TYPE_POINT                  1 << 20

#define DATA_ARRAY              1 << 31

#define DATA_FORMAT_MASK        0x0000ffff
#define DATA_TYPE_MASK          0x00ff0000

#define SIGNATURE_N_CHANNELS(SIG)    (1 + ((SIG >> CV_CN_SHIFT) & CV_MAT_DEPTH_MASK))
#define SIGNATURE_IS_UCHAR(SIG)      ((SIG & CV_MAT_DEPTH_MASK) == CV_8U)
#define SIGNATURE_IS_INT(SIG)        ((SIG & CV_MAT_DEPTH_MASK) == CV_32S)
#define SIGNATURE_IS_FLOAT(SIG)      ((SIG & CV_MAT_DEPTH_MASK) == CV_32F)
#define SIGNATURE_IS_IMAGE(SIG)      (SIG & DATA_TYPE_MASK) == DATA_TYPE_IMAGE
#define SIGNATURE_IS_VECTOR(SIG)     (SIG & DATA_TYPE_MASK) == DATA_TYPE_VECTOR
#define SIGNATURE_IS_POINT(SIG)      (SIG & DATA_TYPE_MASK) == DATA_TYPE_POINT
#define SIGNATURE_IS_WSCALAR(SIG)    (SIG & DATA_TYPE_MASK) == DATA_TYPE_WEIGHTED_SCALAR
#define SIGNATURE_IS_SCALAR(SIG)     (SIG & DATA_TYPE_MASK) == DATA_TYPE_SCALAR


#define DISPLAY_WIN_NAME        "VisionData"

/* DATA FORMAT (OpenCV Mat type):
CV_8UC1:  0   is: 0000000000000000
CV_8UC3:  16  is: 0000000000010000
CV_32FC1: 5   is: 0000000000000101
CV_32FC3: 21  is: 0000000000010101
CV_32SC1: 4   is: 0000000000000100
CV_32SC3: 20  is: 0000000000010100
 */

using namespace std;
using namespace cv;

class CVisionData
{
private:
	int data_signature_; // data signature is a combination of data type(s) and format (and has a flag if its an array)
	CVisionData();

	Mat mat_;
	int internal_row_cnt_;

public:
	CVisionData(const Mat& data, const int& type) : mat_(data), internal_row_cnt_(0)
    {
		data_signature_ = type | mat_.type();
    };

	CVisionData(const CVisionData& other)
	{
		mat_ = other.mat_;
		internal_row_cnt_ = other.internal_row_cnt_;
		data_signature_ = other.data_signature_;
	}

	~CVisionData() {  };

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

	void concatenateColumnwise(const CVisionData& other)
	{
		hconcat(mat_, other.data(), mat_);
	}

	void concatenateRowwise(const CVisionData& other)
	{
		vconcat(mat_, other.data(), mat_);
	}

//	void pushbackRow(const CVisionData& other)
//	{
//		if(other.data().rows + internal_row_cnt_ > mat_.rows)
//		{
//			cerr << "ERROR: Row Pushback not possible, no space left. Call convervativeResizeRows before!" << endl;
//			exit(-2);
//		}
//		other.data().copyTo(mat_.rowRange(internal_row_cnt_, internal_row_cnt_ + other.data().rows));
//		internal_row_cnt_ += other.data().rows;
//	}

	void convervativeResizeRows(const int n_rows)
	{
		Mat temp = Mat::zeros(n_rows, mat_.cols, mat_.type());
		mat_.copyTo(temp.rowRange(0, mat_.rows));
		mat_ = temp;
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
		cout << "Data Type is: " << signature2str();

		switch(data_signature_ & DATA_TYPE_MASK)
		{
		case DATA_TYPE_IMAGE:
		{
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

		case DATA_TYPE_VECTOR:
		{
			cout << "Data Vector Show: #elements: " << mat_.cols*mat_.rows << endl;
			if(SIGNATURE_IS_FLOAT(this->data_signature_))
				cout << "First elements: " << mat_.at<float>(0) << ", " << mat_.at<float>(1) << ", " << mat_.at<float>(2) << endl;
			else if(SIGNATURE_IS_INT(this->data_signature_))
				cout << "First elements: " << mat_.at<int>(0) << ", " << mat_.at<int>(1) << ", " << mat_.at<int>(2) << endl;
			else
				cout << "Signature is non float or non integer." << endl;
			break;
		}
		}



	}

	inline string signature2str() const
	{
		return signature2str(this->data_signature_);
	}

	static string signature2str(const int& signature)
	{
		string r;

		uchar depth = signature & CV_MAT_DEPTH_MASK;
		uchar chans = 1 + ((signature & DATA_FORMAT_MASK) >> CV_CN_SHIFT);

		switch (depth)
		{
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

		switch (signature & DATA_TYPE_MASK)
		{
		case DATA_TYPE_SCALAR:       r += "Scalar"; break;
		case DATA_TYPE_VECTOR:       r += "Vector"; break;
		case DATA_TYPE_IMAGE:        r += "Image"; break;
		case DATA_TYPE_POINT:        r += "Point"; break;
		default:                     r += "Unknown"; break;
		}

		if( signature & DATA_ARRAY )
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

	inline int getSignature() const
	{
		return data_signature_;// & (DATA_TYPE_MASK | DATA_FORMAT_MASK);
	}
};


// DATA CONVERTER Class for automatic data conversion before Vision Modules execute their task

class CVisionDataConverter
{
private:
	CVisionDataConverter();
	vector<void(*)(CVisionData& data)> convert_routines_;
	int expected_input_signature_;

	static void convImage2Vector(CVisionData& data)
	{
		data.assignData(data.data().reshape(0,1), DATA_TYPE_VECTOR);
	}
	static void convScalar2Vector(CVisionData& data)
	{
		data.assignData(data.data().reshape(0,1), DATA_TYPE_VECTOR);
	}
	static void convRGB2Gray(CVisionData& data)
	{
		Mat temp;
		cvtColor(data.data(), temp, CV_BGR2GRAY);
		data.assignData(temp, data.getType());
	}
	static void convUChar2Float(CVisionData& data)
	{
		Mat temp;
		data.data().convertTo(temp, CV_32F, 1.0/255.0);
		data.assignData(temp, data.getType());
	}

public:
	CVisionDataConverter(const int& input_data_signature, const int& output_data_signature) : expected_input_signature_(input_data_signature)
	{
		int converted_signature = input_data_signature;
		if(input_data_signature != output_data_signature)
		{
			// Possible FORMAT conversions
			// ---------------------------

			// RGB values to Grayscale (3 channel -> 1 channel)
			if(SIGNATURE_N_CHANNELS(input_data_signature) == 3 && SIGNATURE_N_CHANNELS(output_data_signature) == 1)
			{
				convert_routines_.push_back(&convRGB2Gray);
				converted_signature &= ~(CV_MAT_TYPE_MASK << CV_CN_SHIFT); // (remove all bits at 0b111000)
			}

			// 8bit to floating point (with range scaling from [0..255] -> [0..1])
			if(SIGNATURE_IS_UCHAR(input_data_signature) && SIGNATURE_IS_FLOAT(output_data_signature))
			{
				convert_routines_.push_back(&convUChar2Float);
				converted_signature &= ~CV_MAT_TYPE_MASK;
				converted_signature |= CV_32F;
			}

			// Possible TYPE conversions
			// -------------------------

			// convert image to vector by concatenating all pixels
			if(SIGNATURE_IS_IMAGE(input_data_signature) && SIGNATURE_IS_VECTOR(output_data_signature))
			{
				convert_routines_.push_back(&convImage2Vector);
				converted_signature &= ~DATA_TYPE_MASK;
				converted_signature |= DATA_TYPE_VECTOR;
			}
			// convert Scalar Value to Vector (with only one entry; probably needed for channel merging)
			if(SIGNATURE_IS_SCALAR(input_data_signature) && SIGNATURE_IS_VECTOR(output_data_signature))
			{
				convert_routines_.push_back(&convScalar2Vector);
				converted_signature &= ~DATA_TYPE_MASK;
				converted_signature |= DATA_TYPE_VECTOR;
			}

		}

		//bitset<32> from(input_data_signature), target(output_data_signature), achieved(converted_signature);

//		cout << "Conversion result: From: " << CVisionData::signature2str(input_data_signature) << endl;
//		cout << "                     To: " << CVisionData::signature2str(output_data_signature) << endl;
//		cout << "           was achieved: " << CVisionData::signature2str(converted_signature) << endl;

		if(output_data_signature != converted_signature)
		{
			cout << "No suitable conversion found! Throwing PipeConfigException" << endl;
			exit(-1);
		}
	};

	~CVisionDataConverter() { };

	void convert(CVisionData& data)
	{
		if(data.getSignature() != expected_input_signature_)
		{
			cout << "Are you kidding me?! I expected a " << CVisionData::signature2str(expected_input_signature_) << " but got a " << data.signature2str() <<" You lied!!!" << endl;
			exit(-1);
		}
		vector<void(*)(CVisionData& data)>::const_iterator convert_routines_it;
		for(convert_routines_it = convert_routines_.begin(); convert_routines_it != convert_routines_.end(); ++convert_routines_it)
		{
			(*convert_routines_it)(data);
		}
	}

};

