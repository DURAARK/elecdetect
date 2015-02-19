/*
 * ElecDetec: RandomForest.cpp
 *
 *  Created on: Feb, 2015
 *      Author: Robert Viehauser
 */

#include "RandomForest.h"


CRandomForest::CRandomForest()
{
    // Random Forest Parameters
    uint max_depth = RF_MAX_DEPTH;  // maximum depth of a tree in the forest
    uint num_of_trees_in_the_forest = RF_N_TREES;  // number of trees in the forest
    int termcrit_type = RF_TERM_CRIT_NSAMPLES;     // stops node splitting when reaching a minimum of samples
    uint min_sample_cnt = 10;                      // number of samples at which no splitting is performed (has only effect with RF_TERM_CRIT_NSAMPES)
    bool calc_train_error = true;  // calculate the training error from out-of-bag samples (just for information)
    bool store_true_probs = false; // store true probabililty distributions in the leaf nodes

    rf_params_ = new RF::Params(max_depth, num_of_trees_in_the_forest, termcrit_type, min_sample_cnt, calc_train_error, store_true_probs);
    rf_ = new RF::RandomForest<float, CLASS_LABEL_TYPE>(*rf_params_);
}

CRandomForest::~CRandomForest()
{
    if(rf_)
        delete rf_;
    rf_ = NULL;

	if(rf_params_)
		delete rf_params_;
	rf_params_ = NULL;
}

void CRandomForest::predict(const vector<float>& feature_vec, vector<WeightedLabel>& result)
{
    CLASS_LABEL_TYPE* labels = NULL;
    float* prop = NULL;
    uint prop_length;

    rf_->predict(&feature_vec[0], labels, prop, prop_length, RF::CUMULATIVE_HIST);

    result.clear();
    for(uint j = 0; j < prop_length; ++j)
    {
        WeightedLabel class_prediction;
        class_prediction.label_ = labels[j];
        class_prediction.weight_ = prop[j];
        result.push_back(class_prediction);
    }

    free(labels); labels = NULL;
    free(prop); prop = NULL;
}

// train_data contains for each sample one CVisionData and train_labels a CVisionData-Label
void CRandomForest::train(const vector<vector<float> >& feature_vecs, const vector<CLASS_LABEL_TYPE>& labels)
{
    ELECDETEC_ASSERT((feature_vecs.size() == labels.size()), "RandomForest: Amount of samples is different to amount of labels!");

    cout << "Training RandomForest (" << rf_params_->num_of_trees_in_the_forest_ << " trees). Please be patient... " << flush;

    // prepare training data
    const int n_samples = feature_vecs.size();
    const int n_features = feature_vecs[0].size();

    //cout << "Data dimensions are: samples:" << n_samples << " dimensions:" << n_features << endl;

    RF::RandomForest<float, CLASS_LABEL_TYPE>::TrainData rf_train_data;

    rf_train_data.f_length_ = static_cast<uint>(n_features);
    rf_train_data.nsamples_ = static_cast<uint>(n_samples);

    rf_train_data.x_data_ = Malloc(const float*, n_samples);//(float**)malloc((n_samples)*sizeof(float*));

    // take care of probably not contiuous stored xdata mat:
    for(int sample_cnt = 0; sample_cnt < n_samples; ++sample_cnt)
    {
        rf_train_data.x_data_[sample_cnt] = &(feature_vecs[sample_cnt][0]);
    }
    rf_train_data.y_data_ = &(labels[0]);

    //cout << "Data:" << endl << rf_train_data.x_data_[0][0] << ", " << rf_train_data.x_data_[0][1] << ", " << rf_train_data.x_data_[0][2] << endl;
    //cout << rf_train_data.x_data_[0][n_features-3] << ", " << rf_train_data.x_data_[0][n_features-2] << ", " << rf_train_data.x_data_[0][n_features-1] << endl;

    rf_->train(rf_train_data);

    // cleanup
    free(rf_train_data.x_data_);

    cout << "done." << endl << flush;
}

void CRandomForest::save(FileStorage& fs) const
{
//	cout << "saving RandomForest..." << endl << flush;
    fs << "Params" << "{";
    fs << "max-depth" << (int)rf_params_->max_depth_;
    fs << "min-sample-cnt" << (int)rf_params_->min_sample_cnt_;
    fs << "ntrees" << (int)rf_params_->num_of_trees_in_the_forest_;
    fs << "termcrit" << rf_params_->termcrit_type_;
    fs << "calc-train-error" << rf_params_->calc_train_error_;
    fs << "store-true-probs" << rf_params_->store_real_probs_in_lnodes_;
    fs << "}";
    rf_->save(fs, "Forest");
//	cout << "RandomForest saved." << endl;
}

void CRandomForest::load(FileNode& node)
{
    if(rf_)
        delete rf_;
    if(rf_params_)
        delete rf_params_;

    rf_params_ = new RF::Params;
    FileNode params = node["Params"];
    params["calc-train-error"] >> rf_params_->calc_train_error_;
    int temp_int; // temp int for cast to uint
    params["max-depth"] >> temp_int; rf_params_->max_depth_ = temp_int;
    params["min-sample-cnt"] >> temp_int; rf_params_->min_sample_cnt_ = temp_int;
    params["ntrees"] >> temp_int; rf_params_->num_of_trees_in_the_forest_ = temp_int;
    params["termcrit"] >> rf_params_->termcrit_type_;
    params["store-true-probs"] >> rf_params_->store_real_probs_in_lnodes_;

    rf_ = new RF::RandomForest<float,CLASS_LABEL_TYPE>(*rf_params_);

    FileNode rf_node = node["Forest"];
    rf_->load(rf_node);
}
