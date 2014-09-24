// ElecDetec.cpp : Defines the entry point for the console application.
//


// TODO list:
/*
 * also save and load configuration of vision modules (HoG-parameters, ...)
 * check config-file by CRC checksum
 * test and add the correct exceptions
 * verbose mode?
 */

#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <cstring>

#include "PipelineController.h"
#include "Utils.h"
#include "tinyxml2.h"


#define RESULT_DIR_NAME    "results" // without "/" !!!


using namespace std;
using namespace cv;

enum ExecMode {TRAINING, TESTING};

int main(int argc, char* argv[])
{
	//------------------------------------------------
	// Reading input parameters
	CommandParams params;
	try
	{
		parseCmd(argc, argv, params);
	}
	catch(ParamExecption& e)
	{
		cout << e.what() << endl;
		exit(-1);
	}
	ExecMode exec_mode = params.str_classifier_.empty() ? TESTING : TRAINING;

	//------------------------------------------------
	// Vision Pipeline
	CPipelineController* pipe = new CPipelineController();

	try
	{
		if(exec_mode ==  TESTING) // on testing, read pipeline configuration from configfile
		{
			cout << "Executing TESTING mode" << endl << flush;
			// initialize/load pipe from config file
			pipe->load(params.str_configfile_);

			// test files in directory
			vector<string> testfiles;
			getFileList(params.str_imgset_, testfiles);

			MKDIR((params.str_imgset_ + RESULT_DIR_NAME).c_str());

			for(vector<string>::const_iterator f_it = testfiles.begin(); f_it != testfiles.end(); ++f_it)
			{
				cout << "---" << endl << flush;
				cout << "Processing File: " << *f_it << endl << flush;
				Mat image = cv::imread(params.str_imgset_ + *f_it);
				vector<vector<Rect> > bb_results; // first dimension: obj-classes, second: objects
				pipe->test(image, bb_results);
				int obj_cnt = 0;
				for(vector<vector<Rect> >::const_iterator class_it = bb_results.begin(); class_it != bb_results.end(); ++class_it)
					obj_cnt += class_it->size();
				cout << "Found " << obj_cnt << " objects." << endl;

				// show and export results
				tinyxml2::XMLDocument xml_doc;
				xml_doc.InsertEndChild(xml_doc.NewDeclaration(NULL));
				tinyxml2::XMLElement* xml_image = xml_doc.NewElement("Image");
				xml_image->SetAttribute("file", f_it->c_str());
				vector<vector<Scalar> > colors = getColors(2);
				for(vector<vector<Rect> >::const_iterator class_it = bb_results.begin(); class_it != bb_results.end(); ++class_it)
				{
					string obj_type = "unknown";
					if(class_it - bb_results.begin() == 0)
						obj_type = "sockets";
					if(class_it - bb_results.begin() == 1)
						obj_type = "switches";
					tinyxml2::XMLElement* xml_object = xml_doc.NewElement(obj_type.c_str());
					for(vector<Rect>::const_iterator rect_it = class_it->begin(); rect_it != class_it->end(); ++rect_it)
					{
						tinyxml2::XMLElement* bounding_box = xml_doc.NewElement("bounding-box");
						stringstream str_x; str_x << rect_it->x;
						stringstream str_y; str_y << rect_it->y;
						stringstream str_w; str_w << rect_it->width;
						stringstream str_h; str_h << rect_it->height;

						bounding_box->SetAttribute("x", str_x.str().c_str());
						bounding_box->SetAttribute("y", str_y.str().c_str());
						bounding_box->SetAttribute("w", str_w.str().c_str());
						bounding_box->SetAttribute("h", str_h.str().c_str());

						xml_object->InsertEndChild(bounding_box);

						rectangle(image, *rect_it, colors[class_it - bb_results.begin()][0], 2);
						cout << "Found object bounded by: " << *rect_it << " with label: \"" << obj_type << "\"" << endl;
					}
					xml_image->InsertEndChild(xml_object);
				}
#ifdef VERBOSE
				imshow("RESULT", image);
#endif
				// get filename prefix
				size_t extention_pos_png = f_it->find(".png"); // search for .png string
				size_t extention_pos_jpg = f_it->find(".jpg"); // search for .jpg string

				size_t extention_pos = f_it->size(); // just append on default
				string str_extention;
				if(extention_pos_png == f_it->size()-4) // ".png" was found at the end
				{
					extention_pos = extention_pos_png;
					str_extention = ".png";
				}
				else if(extention_pos_jpg == f_it->size()-4) // ".jpg" was found at the end
				{
					extention_pos = extention_pos_jpg;
					str_extention = ".jpg";
				}

				string str_filename_prefix = f_it->substr(0, extention_pos);

				string xml_filename = params.str_imgset_ + RESULT_DIR_NAME + FOLDER_CHAR + str_filename_prefix + FILENAME_RESULT_POSTFIX + ".xml";
				string img_filename = params.str_imgset_ + RESULT_DIR_NAME + FOLDER_CHAR + str_filename_prefix + FILENAME_RESULT_POSTFIX + str_extention;

				// write detection results to XML file
				xml_doc.InsertEndChild(xml_image);
				xml_doc.SaveFile(xml_filename.c_str());

				// save output image
				imwrite(img_filename.c_str(), image);

#ifdef VERBOSE
				waitKey(1000);
#endif
			}

		}

		if(exec_mode ==  TRAINING) // TRAINING mode
		{
			cout << "Executing TRAINING mode" << endl << flush;
			// initialize new pipe and train it
			pipe->train(params);

			// save pipe
			pipe->save(params.str_configfile_);
			cout << params.str_configfile_ << " saved." << endl;
		}

	}
	catch(PipeConfigExecption& e)
	{
		cout << e.what() << endl;
		delete pipe; pipe = NULL;
		exit(-1);
	}

	// cleanup
#ifdef VERBOSE
	cv::destroyAllWindows();
#endif

	delete pipe; pipe = NULL;

	return 0;
}







