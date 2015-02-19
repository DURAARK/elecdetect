/*
 * CAnnotation.h
 *
 *  Created on: Jun 13, 2014
 *      Author: test
 */

#ifndef CANNOTATION_H_
#define CANNOTATION_H_

#include <string>
#include <vector>
#include <iomanip>
#include <sstream>

#include <opencv2/opencv.hpp>

#include "tinyxml2.h"
#include "Utils.h"


#define SMP_FULL_SIZE_MM     128.0 // how big is the whole sample incl. context
#define SOCKET_DIAMETER_MM    38.0 // socket diameter that will be matched for socket annotations
#define PIXEL_PER_SQ_MM        1.0 // spatial resolution of the samples
#define SWITCH_SIZE_MM        55.0 // reference switch size
#define SWITCH_SIZE_MM_FROM   50.0 // generate different sizes of switches (FROM-TO)
#define SWITCH_SIZE_MM_TO     60.0
#define POS_SMP_PER_ANNO       5   // how many random samples are generated from one annotation
#define NEG_SMP_PER_IMG       10   // how many negative examples are extracted from one image

#define IMG_VIEW_SIZE 1024
#define IMG_ZOOM_REGION 300

using namespace std;
using namespace cv;
using namespace tinyxml2;

void onMouse(int event, int x, int y, int flag, void* gui_ptr_arg);

class CAnnotation {
private:
    CAnnotation() : img_filename_(""), exit_annotation_(false), rescale_view_factor_(0) { }

	string img_filename_;
	Mat img_;

	vector<vector<Point> > sockets_;
	vector<vector<Point> > switches_;
	vector<Point> rect_plane_;

	vector<Point> cur_annotation_;

	bool exit_annotation_;

	double rescale_view_factor_;

	static int crop_socket_cnt_;
	static int crop_switch_cnt_;
	static int crop_neg_cnt_;

public:
	// constructor that reads the image
	CAnnotation(string img_filename);

	// copy constructor
	CAnnotation(const CAnnotation& other);
	virtual ~CAnnotation();

	bool isImageLoaded();

	// annotate: user interaction method (automatically resets previous annotations)
	bool annotate(); // returns false if the programm needs to be aborted

	// refine existing annotations
	void refine();

	// show: shows image with annotations
	void show();

	// zooming window method
	void zoomRegion(int x, int y);

	// reset annotations
	void reset();

	// write annotations as XML string
	void addXMLString(XMLDocument& doc);

	// load annotations from an XML Element
	void loadFromXMLElement(XMLElement* xml_el);

	// crop image
	void saveAnnotatedImages(const string& prefix = "");

	void mouseInputHandler(int event, int flag, int x, int y);

};

#endif /* CANNOTATION_H_ */
