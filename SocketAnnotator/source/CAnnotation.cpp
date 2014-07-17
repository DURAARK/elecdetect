/*
 * CAnnotation.cpp
 *
 *  Created on: Jun 13, 2014
 *      Author: test
 */

#include "CAnnotation.h"

static void onMouse(int event, int x, int y, int flag, void* gui_ptr_arg);

CAnnotation::CAnnotation(string img_filename) : img_filename_(img_filename), exit_annotation_(false), waiting_for_key_(false)
{

}

CAnnotation::~CAnnotation() {

}

CAnnotation::CAnnotation(const CAnnotation& other) : exit_annotation_(false), waiting_for_key_(false)
{
	this->cur_annotation_ = other.cur_annotation_;
	this->exit_annotation_ = other.exit_annotation_;
	this->img_filename_ = other.img_filename_;
	this->sockets_ = other.sockets_;
	this->switches_ = other.switches_;
}

void CAnnotation::annotate()
{
	// show image
	show();

	// get user input
	setMouseCallback("Annotate Image", &onMouse, this);
	while(!exit_annotation_)
	{
		if(cur_annotation_.size() == 4) // if this was the 4th corner
		{
			cout << "waitfor key" << endl;
			waiting_for_key_ = true;
			char key = waitKey(0);
			waiting_for_key_ = false;
			cout << "got key" << endl;
			if(key == 'x') // on key 'x' -> save this is a socket
			{
				sockets_.push_back(cur_annotation_);
			}
			else if(key == 's') // on key 's' -> save this is a switch
			{
				switches_.push_back(cur_annotation_);
			}
			else if(key == 27) // on key ESC -> reset all annotations
			{
				reset();
			}
			// otherwise reject current annotation
			cur_annotation_.clear();
			show();
		}
		else if(cur_annotation_.size() > 4)
		{
			cur_annotation_.clear();
			show();
		}

		waitKey(100);
	}

	// save user input
	cout << "Annotate of current image finished. Got " << sockets_.size() << " sockets and " << switches_.size() << " switches." << endl;

	//TODO: cut out annotated regions and save the images into subfolders
}

void CAnnotation::show()
{
	// read image
	Mat img0 = imread(img_filename_);
	if(!img0.data)
	{
		std::cerr << "Annotation: File doesn't exist. Something went terribly wrong!" << std::endl;
		exit(-2);
	}
	//imshow("Annotate Image", img);

	Mat img;
	resize(img0, img, Size(1024, (int)((float)img0.rows / ((float)img0.cols / 1024.0))));

	Mat overlay_img = img.clone();

	// draw annotations
	polylines(overlay_img, switches_, true, Scalar(0,0,255), 2);
	polylines(overlay_img, sockets_, true, Scalar(255,0,0), 2);

	overlay_img = overlay_img*0.8 + img*0.2;

	bool close_cur_anno = cur_annotation_.size()==4;
	Scalar cur_anno_color(0, 255, close_cur_anno ? 0 : 255);
	if(!cur_annotation_.empty()) // opencv-bug?
	{
		if(cur_annotation_.size() > 1)
			polylines(overlay_img, cur_annotation_, close_cur_anno, cur_anno_color, 2);
		else
			circle(overlay_img, cur_annotation_[0], 2, cur_anno_color, -1);
	}
	imshow("Annotate Image", overlay_img);

//	cout << "is cur_anno empty? " << cur_annotation_.empty() << ", and what's its size=" << cur_annotation_.size() << endl;

//	vector<vector<Point> >::const_iterator sw_it;
//	for(sw_it = switches_.begin(); sw_it != switches_.end(); ++sw_it)
//	{
//		vector<Point>::const_iterator pt_it;
//		for(pt_it = sw_it->begin(); pt_it != sw_it->end(); ++pt_it)
//		{
//
//		}
//	}


}

void CAnnotation::mouseInputHandler(int event, int flag, int x, int y)
{
	if(!waiting_for_key_)
	{
		if(event == CV_EVENT_RBUTTONDOWN) // finish annotation of the whole image on double click
		{
			cout << "Exit Annotation" << endl;
			exit_annotation_ = true;
			return;
		}
		if(event == CV_EVENT_LBUTTONDOWN)
		{
			cout << "Cur Anno point pushback" << endl;
			cur_annotation_.push_back(Point(x,y));
			show();
		}
	}
}

static void onMouse(int event, int x, int y, int flag, void* gui_ptr_arg)
{
    if(event != CV_EVENT_MOUSEMOVE)
    {
    	CAnnotation* anno_ptr = static_cast<CAnnotation*>(gui_ptr_arg);
        anno_ptr->mouseInputHandler(event, flag, x, y);
    }
}

void CAnnotation::reset()
{
	sockets_.clear();
	switches_.clear();
}
