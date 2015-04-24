/*
 * ElecDetec: Defines.h
 *
 *  Created on: Feb, 2015
 *      Author: Robert Viehauser
 */


#ifndef DEFINES_H_
#define DEFINES_H_

#define CLASS_LABEL_TYPE     int

//! Default Settings from ini file:

// name of the result directory that is created inside the image folder (at detection)
// without folder char "/" or "\" !!!
#define DEFAULT_RESULT_DIRECTORY_NAME      "results"
// Write probability maps?
#define DEFAULT_WRITE_PROBABILITY_MAPS     false
// Suffix of the result files
#define DEFAULT_FILENAME_RESULT_SUFFIX       "-result"
// Suffix of the result probability maps
#define DEFAULT_PROB_MAP_RESULT_SUFFIX       "-probmap"
// Maximum Bounding-Box overlap (junction area / union area)
#define DEFAULT_MAX_BOUNDINGBOX_OVERLAP       0.7
// detection thresholds for the different classes
#define DEFAULT_DETECTION_LABEL_THRESHOLDS         ""           //"0.25, 0.67"
// specifies the assignment from the defined thresholds to the classes
#define DEFAULT_DETECTION_LABELS                   ""           //"   1,    2"
// default threshold for unspecified classes
#define DEFAULT_DETECTION_DEFAULT_THRESHOLD        "0.5"

// defines which character string separates the label from remaining filename of the training patches
#define DEFAULT_LABEL_DELIMITER            "_"
// Number of Bootstrap-Stages
#define DEFAULT_MAX_BOOTSTRAP_STAGES        7

// Sliding Window size in mm. Has to match training patch size too!
#define DEFAULT_PATCH_SIZE                128
// possible image file extentions (codec has to be present in OpenCV)
#define DEFAULT_IMG_FILE_EXTENTIONS        "jpg,jpeg,png"
// Label of the Background class (has to be consistent with the training data)
#define DEAFULT_BACKGROUND_LABEL            0


//! other defines

// Defines the size of the search grid on an image (less: finer grid, slower. higher: coarse grid, faster)
#define OPENING_SIZE                5
// Minimal error on all training samples to continue bootstrapping
#define BOOTSTRAP_MIN_ERROR         3e-05

// XML node names
#define NODE_NAME_ALGORITHM_DATA       "AlgorithmData"
#define NODE_NAME_CLASS_LABELS         "classeslabels"
#define NODE_NAME_PATCH_SIZE           "patchsize"
#define NODE_NAME_BG_LABEL             "backgroundlabel"
#define NODE_NAME_GOF                  "GradientOrientationFeatures"
#define NODE_NAME_ORI_HAAR_FEATURES    "OrientationHaarFeatures"
#define NODE_NAME_COLOR_HAAR_FEATURES  "ColorHaarFeatures"
#define NODE_NAME_RANDOM_FOREST        "RandomForest"

// - SYSTEM DEPENDENT defines
#if defined _WIN32
  #include <conio.h>
  #include <direct.h>
  #define MKDIR(path) _mkdir(path);
  #define FOLDER_CHAR  "\\"
#elif defined __linux__
  #include <sys/types.h>
  #include <sys/stat.h>
  #define MKDIR(path) mkdir(path, 0777); // notice that 777 is different than 0777
  #define FOLDER_CHAR  "/"
#endif
// -------------------------


//- Result Evaluation Settings --------------------------------
// if defined, detection results are suited for precision-recall curves
//#define PERFORM_EVALUATION

#ifdef PERFORM_EVALUATION

#define INTER_CLASS_NON_MAXIMA_SUPPRESSION  false   // remove overlapping detection results also between different classes

//#define THRESHOLDS         "0.20, 0.20"  // detection thresholds for the different classes
//#define THRESHOLD_LABELS   "   1,    2"  // specifies the assignment from the defined thresholds to the classes
//#define DEFAULT_THRESHOLD  "0.5"         // default threshold for unspecified classes

#else

#define INTER_CLASS_NON_MAXIMA_SUPPRESSION  true   // remove overlapping detection results also between different classes

//#define THRESHOLDS         "0.25, 0.67"  // detection thresholds for the different classes
//#define THRESHOLD_LABELS   "   1,    2"  // specifies the assignment from the defined thresholds to the classes
//#define DEFAULT_THRESHOLD  "0.5"         // default threshold for unspecified classes

#endif
//------------------------------------------------------------


#endif /* DEFINES_H_ */
