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

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;



class CAnnotation {
private:
	string img_filename_;
	vector<vector<Point> > sockets_;
	vector<vector<Point> > switches_;

	vector<Point> cur_annotation_;

	bool exit_annotation_;
	bool waiting_for_key_;

public:
	CAnnotation(string img_filename);
	CAnnotation(const CAnnotation& other);
	virtual ~CAnnotation();

	//TODO: copy constructor

	// annotate: user interaction method (automatically resets previous annotations)
	void annotate();

	// show: shows image with annotations
	void show();

	// reset annoations
	void reset();

	void mouseInputHandler(int event, int flag, int x, int y);

};

#endif /* CANNOTATION_H_ */
