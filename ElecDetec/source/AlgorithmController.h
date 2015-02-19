/*
 * ElecDetec: AlgorithmController.h
 *
 *  Created on: Feb, 2015
 *      Author: Robert Viehauser
 */

#ifndef PIPELINECONTROLLER_H_
#define PIPELINECONTROLLER_H_

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <bitset>
#include <iomanip>
#include <omp.h>

#include <Defines.h>
#include <Utils.h>

#include <OrientationFilter.h>
#include <HaarWavelets.h>
#include <GradientOrientationFeatures.h>
#include <Hog.h>

#include <RandomForest.h>

using namespace std;
using namespace cv;

extern CLASS_LABEL_TYPE _BACKGROUND_LABEL_;
extern int _MAX_BOOTSTRAP_STAGES_;
extern string _LABEL_DELIMITER_;



// This Exception is thrown if command parameter are faulty
class FileAccessExecption : public exception
{
private:
    const char* reason_;
public:
    //ParamExecption(){ /*reason_ = "no reason";*/ };
    FileAccessExecption(const char* reason) : reason_(reason) { }

    virtual const char* what() const throw()
    {
        stringstream msg;
        msg << "Exception happened! Reason: " << reason_ << endl;
        return msg.str().c_str();
    }
};


/* --------------------------
 * Algorithm Controller Class
 * --------------------------
 */
class CAlgorithmController
{
private:
    // --- Algorithms ---
    // Gradient direction filter
    COrientationFilter* filter_horiz_;
    COrientationFilter* filter_vert_;
    // FeatureExtractors:
    CHog* hog_extractor_;
    CHaarWavelets* ori_haar_extractor_;
    CHaarWavelets* haar_extractor_;
    CGradientOrientationFeatures* gof_extractor_;
    // Classifier
    CRandomForest* random_forest_;
    // ------------------

    // class label mapping and backward mapping to its index
    vector<CLASS_LABEL_TYPE> class_labels_;

    inline int classLabel2Idx(const CLASS_LABEL_TYPE& label)
    {
        vector<CLASS_LABEL_TYPE>::const_iterator cl_it;
        for(cl_it = class_labels_.begin(); cl_it != class_labels_.end(); ++cl_it)
            if(label == *cl_it)
                break;

        if(cl_it == class_labels_.end())
            return -1;

        return distance<vector<CLASS_LABEL_TYPE>::const_iterator>(class_labels_.begin(), cl_it);
    }

    // required for bootstrapping
    int bootstrap_nstages_;
    float bootstrap_train_error_threshold_;
    void bootstrap_getRandomlySelectedSampleIndices(const vector<CLASS_LABEL_TYPE>& data_labels, vector<uint>& selected_sample_indices);

    // extractFeatures
    void extractFeatures(const Mat& img_patch, vector<float>& feature_vec);

    void clearAlgorithm();
    void reInitAlgorithm();

public:
    CAlgorithmController();

    ~CAlgorithmController();

    //void test(const Mat& input_img, vector<vector<Rect> >& bb_results);
    void detect(const Mat& input_img, const string& image_name, vector<Mat>& result_probs, vector<CLASS_LABEL_TYPE>& labels) throw(FileAccessExecption);

    void train(const vector<string>& trainfiles) throw(FileAccessExecption);

    void loadAlgorithmData(const string& filename) throw(FileAccessExecption);

    void saveAlgorithmData(const string& filename) const throw(FileAccessExecption);


};

#endif /* PIPELINECONTROLLER_H_ */
