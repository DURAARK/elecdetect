/*
 * PipelineController.h
 *
 *  Created on: Jul 3, 2014
 *      Author: test
 */

#ifndef PIPELINECONTROLLER_H_
#define PIPELINECONTROLLER_H_


#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
//#include <omp.h>

#include "VisionModule.h"
#include "VectorArray.h"
#include "Scalar.h"
#include "BinaryContours.h"
#include "GradientImage.h"
#include "DistanceTransform.h"
#include "Hog.h"
#include "OwnBrief.h"
#include "DummyFeature.h"
#include "PCA.h"
#include "SVM.h"
#include "RandomForest.h"
#include "Utils.h"

#define CONFIG_NAME_PREPROC              "preprocessing-module"
#define CONFIG_NAME_FEATURE              "feature-module"
#define CONFIG_NAME_CLASSIFIER           "classifier-module"
#define CONFIG_NAME_NUM_CLASSES          "number-of-classes"

#define FILENAME_RESULT_POSTFIX      "-result"

#define ID_CANNY     "canny"
#define ID_GRADIENT  "grad"
#define ID_DISTTR    "dist"
#define ID_HOG       "hog"
#define ID_BRIEF     "brief"
#define ID_SVM       "svm"
#define ID_RF        "rf"

#define SWIN_SIZE     86 // Sliding Window size 90 mm
#define OPENING_SIZE  5  // Kernel Size of morphological opening in non-weihgted results

using namespace std;
using namespace cv;

extern bool getFileList(string directory, vector<string>& filelist);

class CLabeledWeightedRect
{
private:
	inline CLabeledWeightedRect() : weight_(0), label_(0) { };
public:
	inline CLabeledWeightedRect(const Rect& rect, const float& weight, const int& label)
	{
		rect_ = rect;
		weight_ = weight;
		label_ = label;
	}
	inline ~CLabeledWeightedRect() { }
	Rect rect_;
	float weight_;
	int label_;
	inline float getOverlapWith(const CLabeledWeightedRect& other_rect) const
	{
		return static_cast<float>((this->rect_ & other_rect.rect_).area()) / static_cast<float>((this->rect_ | other_rect.rect_).area());
	}
};

inline bool greaterLabeledWeightedRect (const CLabeledWeightedRect& i,const CLabeledWeightedRect& j) { return (i.weight_ > j.weight_); }

class CPipelineController
{
public:
	struct Params
	{
		vector<string> vec_preproc_, vec_feature_;
		string str_classifier_;
	};

private:

	vector<CVisionModule*> v_modules_;
	Params params_;
	int feature_length_; // holds training data sizes of modules with require all samples at once for training
	int n_object_classes_;

	// initializes new untrained pipeline according to params_
	void initializeFromParameters() throw (PipeConfigExecption);
	void deletePipe();

	void postProcessResults(const Mat& labels, const Mat& probability, vector<vector<Rect> >& results);

public:
	CPipelineController();
	virtual ~CPipelineController();

	void test(const Mat& input_img, vector<vector<Rect> >& bb_results);

	void train(const CommandParams& params);

	void printConfig();

	void load(const string& filename);

	void save(const string& filename);


};

#endif /* PIPELINECONTROLLER_H_ */
