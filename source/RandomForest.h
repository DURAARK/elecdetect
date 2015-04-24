/*
 * ElecDetec: RandomForest.h
 *
 *  Created on: Feb, 2015
 *      Author: Robert Viehauser
 */

#ifndef RANDOMFOREST_H_
#define RANDOMFOREST_H_

#include <opencv2/opencv.hpp>
#include <time.h>
#include <fstream>

#include <rf/RandomForest.h>
#include <rf/Params.h>

#include <Defines.h>
#include <Utils.h>

// Random Forest Parameter:
#define RF_N_TREES         250
#define RF_MAX_DEPTH        22

using namespace std;
using namespace cv;

class CRandomForest
{
public:
    struct WeightedLabel
    {
        CLASS_LABEL_TYPE label_;
        float weight_;
    };

private:

    RF::RandomForest<float,int>* rf_;
    RF::Params* rf_params_;

public:
    CRandomForest();
	virtual ~CRandomForest();

    void predict(const vector<float>& feature_vec, vector<WeightedLabel>& result);
    void train(const vector<vector<float> >& feature_vecs, const vector<CLASS_LABEL_TYPE>& labels);
	void save(FileStorage& fs) const;
    void load(FileNode& node);
};

#endif /* RANDOMFOREST_H_ */
