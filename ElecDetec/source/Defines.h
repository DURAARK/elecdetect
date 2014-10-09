/*
 * Defines.h
 *
 *  Created on: Oct 6, 2014
 *      Author: test
 */

#ifndef DEFINES_H_
#define DEFINES_H_

#define VERBOSE  // enable verbose mode

// root XML element names in the created config file
#define CONFIG_NAME_CHANNEL              "feature-channel"
//#define CONFIG_NAME_CHANNEL_LENGTHS      "feature-channel-lengths"
#define CONFIG_NAME_CLASSIFIER           "classifier-module"
#define CONFIG_NAME_NUM_CLASSES          "number-of-classes"
#define CONFIG_NAME_SWIN_SIZE            "sw-size"

// Postfix of the result files
#define FILENAME_RESULT_POSTFIX      "-result"

// Module parameter conventions
#define MODULE_PARAM_PRELUDE    "(" // MUST NOT CONTAIN A SPACE " "
#define MODULE_PARAM_ENDING     ")" // MUST NOT CONTAIN A SPACE " "
#define MODULE_PARAM_DELIMITER  ":" // MUST NOT CONTAIN A SPACE " "
#define ID_CANNY     "canny"
#define ID_GRADIENT  "grad"
#define ID_QGRAD     "qgrad"
#define ID_ORI       "ori"
#define ID_CCHAN     "color"
#define ID_DISTTR    "dist"
#define ID_GOF       "gof"
#define ID_HOG       "hog"
#define ID_BRIEF     "brief"
#define ID_HAAR_FEAT "haar"
#define ID_PCA       "pca"
#define ID_SVM       "svm"
#define ID_LIN_SVM   "linsvm"
#define ID_RF        "rf"

#define SWIN_SIZE                128  //! Change for new testset to 96! also in Hog.cpp!! Sliding Window size in mm
#define OPENING_SIZE               3  // Kernel Size of morphological opening in non-weihgted results
#define MAX_NUMBER_BG_SAMPLES  25000  // maximum number of background samples to prevent memory overflow

// Module specific defines:

// BRIEF:
#define BRIEF_FEATURE_LENGTH 1024
#define BRIEF_CONFIG_NAME_TESTPAIRS   "brief-testpairs"

// Haar-Like Features:
#define HAAR_FEAT_DEFAULT_N_TESTS                        1024  // Number of default testpairs
#define HAAR_FEAT_DEFAULT_SYM_PERCENT                       0  // Percentage of forced symmetric arranged testpairs
#define HAAR_FEAT_MIN_RECT_SZ                               2
#define HAAR_FEAT_MAX_RECT_SZ                              32
#define HAAR_FEAT_CONFIG_NAME            "haar-like-features"
#define HAAR_FEAT_CONFIG_NAME_TESTPAIRS     "haar-feat-rects"

// Gradient Orientation Features:
#define GOF_DEFAULT_N_TESTS                                    1024  // Number of default testpairs
#define GOF_DEFAULT_SYM_PERCENT                                  30  // Percentage of forced symmetric arranged testpairs
#define GOF_MIN_RECT_SZ                                           2
#define GOF_MAX_RECT_SZ                                          32
#define GOF_CONFIG_NAME              "gradient-orientation-features"
#define GOF_CONFIG_NAME_TESTPAIRS       "gradient-orientation-rects"

// Orientation Filter:
#define ORIENTATION_DEFAULT_N_BINS                          5
#define ORIENTATION_DEFAULT_BIN_IDX                         0
#define ORIENTATION_CONFIG_NAME          "orientation-filter"
#define ORIENTATION_CONFIG_NAME_N_BINS               "n-bins"
#define ORIENTATION_CONFIG_NAME_BIN_IDX             "bin-idx"

// Color Channel Filter:
#define COLOR_CHANNEL_DEFAULT_CHANNEL                     "R"
#define COLOR_CHANNEL_CONFIG_NAME      "color-channel-filter"


#endif /* DEFINES_H_ */
