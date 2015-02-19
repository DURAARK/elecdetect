/*
 * ElecDetec: Debug.h
 *
 *  Created on: Feb, 2015
 *      Author: Robert Viehauser
 */

#ifndef DEBUG_H_
#define DEBUG_H_

#include <stdio.h>
#include <iostream>

#define DEBUG_COUT(number) \
	std::cout << "DEBUG:" << number << std::endl << std::flush;

#define DEBUG_COUT_VAR(var) \
    std::cout << #var << " : " << var << std::endl << std::flush;

#define PAUSE_AND_SHOW(image) \
    std::cout << "  Paused - press any key" << std::endl; \
    cv::namedWindow("DEBUG"); \
    cv::imshow("DEBUG", image); \
    cv::waitKey(); \
    cv::destroyWindow("DEBUG");


#endif /* DEBUG_H_ */
