/*
 * Debug.h
 *
 *  Created on: Aug 22, 2014
 *      Author: test
 */

#ifndef DEBUG_H_
#define DEBUG_H_


#define PAUSE_AND_SHOW(image) \
    std::cout << "  Paused - press any key" << std::endl; \
    cv::namedWindow("DEBUG"); \
    cv::imshow("DEBUG", image); \
    cv::waitKey(); \
    cv::destroyWindow("DEBUG");


#endif /* DEBUG_H_ */
