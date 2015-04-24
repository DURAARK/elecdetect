/*
 * ElecDetec: ElecDetec.h
 *
 *  Created on: Feb, 2015
 *      Author: Robert Viehauser
 */


#ifndef ELECDETEC_H
#define ELECDETEC_H


#include <exception>
#include <sstream>
#include <opencv2/opencv.hpp>
#ifdef _MSC_VER
  #include "msvc_dirent.h"
#else
  #include <dirent.h>
#endif

#include <tclap/CmdLine.h>
#include <tinyxml2.h>
#include <AlgorithmController.h>

#include <Debug.h>

using namespace std;
using namespace cv;

extern int _PATCH_WINDOW_SIZE_;
extern string _IMG_FILE_EXTENTIONS_;
extern string _RESULT_DIRECTORY_NAME_;
extern string _PROB_MAP_RESULT_SUFFIX_;
extern string _DETECTION_LABEL_THRESHOLDS_;
extern string _DETECTION_LABELS_;
extern string _DETECTION_DEFAULT_THRESHOLD_;
extern string _FILENAME_RESULT_SUFFIX_;
extern float _MAX_BOUNDINGBOX_OVERLAP_;
extern bool _WRITE_PROBABILITY_MAPS_;


class CLabeledWeightedRect
{
private:
    CLabeledWeightedRect();
public:
    CLabeledWeightedRect(const Rect& rect, const float& weight, const CLASS_LABEL_TYPE& label)
    {
        rect_ = rect;
        weight_ = weight;
        label_ = label;
    }
    CLabeledWeightedRect(const CLabeledWeightedRect& other)
    {
        rect_ = other.rect_;
        weight_ = other.weight_;
        label_ = other.label_;
    }
    ~CLabeledWeightedRect() { }

    Rect rect_;
    float weight_;
    CLASS_LABEL_TYPE label_;

    inline float getOverlapWith(const CLabeledWeightedRect& other_rect) const
    {
        return static_cast<float>((this->rect_ & other_rect.rect_).area()) / static_cast<float>((this->rect_ | other_rect.rect_).area());
    }

    static bool greaterWeight(const CLabeledWeightedRect& i,const CLabeledWeightedRect& j)
    {
        return (i.weight_ > j.weight_);
    }
};


class ElecDetec
{
public:
    struct ExecutionParameter
    {
        enum ExecMode {TRAIN, DETECT};
        ExecMode exec_mode_;
        string str_imgset_, str_configfile_;
    };

    // is thrown if command parameter are faulty
    class ExecParamExecption : public exception
    {
    private:
        const char* reason_;
    public:
        ExecParamExecption(const char* reason) : reason_(reason) { }

        virtual const char* what() const throw()
        {
            stringstream msg;
            msg << "Exception happened! Reason: " << reason_ << endl;
            return msg.str().c_str();
        }
    };

private:
    ElecDetec();
    ExecutionParameter exec_params_;

    void getImgsFromDir(string directory, vector<string> &filelist) throw(ExecParamExecption);

    void postProcessResultImages(const Mat& original_image, const vector<Mat>& prob_imgs,
                                 const vector<CLASS_LABEL_TYPE>& labels, const string& input_filename, const string& root_folder);
    void evaluateProbImages(const vector<Mat>& prob_imgs, const vector<CLASS_LABEL_TYPE>& labels,
                            vector<CLabeledWeightedRect>& result_bboxes, map<CLASS_LABEL_TYPE, float>& threshold_map);
    void nonMaximaSuppression(vector<CLabeledWeightedRect>& candidates, vector<int>& max_indices);
    void writeResultImage(const Mat& original_image, const vector<CLabeledWeightedRect>& results, const string& output_filename);
    vector<tinyxml2::XMLElement*> getDetectionXMLNodes(tinyxml2::XMLDocument& xml_doc, vector<CLabeledWeightedRect>& result_bboxes);

public:
    ElecDetec(const ExecutionParameter& exec_params) throw(ExecParamExecption);
    void doAction() throw(ExecParamExecption);

};







#endif // ELECDETEC_H
