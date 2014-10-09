/*
 * Debug.h
 *
 *  Created on: Aug 22, 2014
 *      Author: test
 */

#ifndef DEBUG_H_
#define DEBUG_H_

#include <stdio.h>
#include <iostream>

#define DEBUG_COUT(number) \
	std::cout << "DEBUG:" << number << std::endl << std::flush;

#define PAUSE_AND_SHOW(image) \
    std::cout << "  Paused - press any key" << std::endl; \
    cv::namedWindow("DEBUG"); \
    cv::imshow("DEBUG", image); \
    cv::waitKey(); \
    cv::destroyWindow("DEBUG");


#endif /* DEBUG_H_ */
