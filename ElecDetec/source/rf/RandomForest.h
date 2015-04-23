/*
 * RandomForest Implementation
 * RandomForest.h
 *
 *  Created on: Okt, 2014
 *      Author: Robert Viehauser
 *              robert(dot)viehauser(at)gmail(dot)com
 */


#ifndef RANDOMFOREST_H
#define RANDOMFOREST_H


#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <cassert>
#include "Params.h"

//#define RF_VERBOSE                     // enables verbose mode: print training status
//#define RF_VERBOSE_PRINT_TRAIN_ERROR   // enables printing the train error after training

// split definitions:
#define RF_NSAMPLES_NODE_SPLIT_BAGGING_FUNC(x)   max(static_cast<uint>(x/3), (uint)30); //tree_->forest_->params_.min_sample_cnt_);
#define RF_NCHOSEN_VARS_PER_SPLIT_NODE_FUNC(x)   sqrt(x)
#define RF_NTRIED_THRESHOLDS_PER_SPLIT_VARIABLE  10


#define FREE_AND_NULL(ptr) free(ptr); \
                           ptr = NULL;

using namespace std;

namespace RF
{
// Enum type for the user to specify the return value from the predict-method.
// SINGLE_CLASS_LABEL: The YDataType* result will be set to the predicted class label (length is 1),
// calculated by the max of the accumulated histogram of all trees. The float* prop will be set to NULL
// SINGLE_CLASS_LABEL_PROP: The YDataType result pointer will be set to the predicted class label containing the value as
// described above, and the prob pointer will be set to a float confidence value (= max value of the normalized accumulated histogram)
// CUMULATIVE_HIST: The YDataType result pointer is set to an array of class labels. The prop pointer will point at a float vector of nclasses_ length
// containing all data of the accumulated histogram (is normalized). Both vectors are coherent, i.e. a label with probability share the same index.
// TREE_HISTS: The result value will not be set and the prop pointer points at an array of length ntrees_*nclasses
// containing all histograms of the trees leaf nodes. This type is often used for classifier stacking (are not normalized if store_ture_probs was not set to true
// during training)
enum PredictReturnType { SINGLE_CLASS_LABEL, SINGLE_CLASS_LABEL_PROP, CUMULATIVE_HIST, TREE_HISTS };

template<class XDataType, class YDataType>
class RandomForest
{
public:
    // Struct train data: for the user to create a train data struct
    struct TrainData
    {
        const XDataType** x_data_; // data pointer to X-Data. First dimension is the sample (x_data_[3][5] = sample3-feature5, sample1-feature2, .., sample1-feature(f_length) , sample2-feature1, ...)
        const YDataType* y_data_; // data pointer to Y_Data (Array of data_length)
        uint nsamples_;     // number of training samples data (
        uint f_length_;     // length of one X-Data sample, feature-length

        TrainData(XDataType** x_data_ptr, YDataType* y_data_ptr, const uint& nsamples, const uint& feature_length) :
            x_data_(x_data_ptr), y_data_(y_data_ptr), nsamples_(nsamples), f_length_(feature_length)
        {

        }
        TrainData() : x_data_(NULL), y_data_(NULL), nsamples_(0), f_length_(0)
        {

        }
    };



private:
    class Node;
    class RandomTree;

    // private Node: not for users
    class Node
    {
    private:
        RandomTree* tree_;
        enum NodeType {SPLIT, LEAF};
        NodeType node_type_;
        uint depth_;

        Node* l_;
        Node* r_;

        float* leaf_dist_;

        struct Split
        {
            uint var_idx_;
            XDataType var_threshold_;
            float score_;

            Split(uint var_idx, XDataType var_th, float score) : var_idx_(var_idx), var_threshold_(var_th), score_(score)
            {

            }
        } split_;

        // calculates float* distribution of length nclasses_ for given YData and subset mask (direct indexed)
        float* const calcWeightedClassProbabilitiesForYData(const YDataType* const y_data, const uint* const subset_indices_vec, const uint& subset_indices_length) const
        {
            const uint nclasses = tree_->forest_->nclasses_;
            const float* const class_weights = tree_->forest_->class_weights_;
            float* const y_prob = Malloc(float, nclasses);
            memset(y_prob, 0.0, sizeof(float)*nclasses);
            float l_sum = 0.0;
            for(uint i = 0; i < subset_indices_length; ++i)
            {
                for(uint j = 0; j < nclasses; ++j)
                {
                    if(y_data[subset_indices_vec[i]] == tree_->forest_->sorted_class_labels_[j])
                    {
                        y_prob[j] += class_weights[j];
                        l_sum += class_weights[j];
                        break;
                    }
                }
            }
            for(uint j = 0; j < nclasses; ++j)
            {
                y_prob[j] = y_prob[j]/l_sum;
            }
            return y_prob;
        }

        // calculates float* histogram of weighted YData samples
        float* const calcWeightedClassHistogramForYData(const YDataType* const y_data, const uint* subset_indices_vec, const uint& subset_indices_length) const
        {
            const uint nclasses = tree_->forest_->nclasses_;
            const float* const class_weights = tree_->forest_->class_weights_;
            float* const y_hist = Malloc(float, nclasses);
            memset(y_hist, 0.0, sizeof(float)*nclasses);
            for(uint i = 0; i < subset_indices_length; ++i)
            {
                for(uint j = 0; j < nclasses; ++j)
                {
                    if(y_data[subset_indices_vec[i]] == tree_->forest_->sorted_class_labels_[j])
                    {
                        y_hist[j] += class_weights[j];
                        break;
                    }
                }
            }
            return y_hist;
        }

        const float getEntropy(const YDataType* const input_vec, const uint* const subset_indices_vec, const uint& subset_indices_length) const
        {
            float* const label_prob = calcWeightedClassProbabilitiesForYData(input_vec, subset_indices_vec, subset_indices_length);
            const uint nclasses = tree_->forest_->nclasses_;

            float entropy = 0;
            for(uint j = 0; j < nclasses; ++j)
            {
                if(label_prob[j] > 0.0) // 0*log2(0) = 0
                {
                    const float p = label_prob[j];
                    entropy -= p * log2(p);
                }
            }
            free(label_prob);
            return entropy;
        }

        inline const float informationGainForSplit(const TrainData& train_data, const uint* const all_idxes_vec, const uint& all_idxes_vec_length,
                                                   const uint* const l_idxes, const uint& l_idxes_length, const uint* const r_idxes, const uint& r_idxes_length)
        {
            float entropy_all = getEntropy(train_data.y_data_, all_idxes_vec, all_idxes_vec_length);
            float entropy_left = getEntropy(train_data.y_data_, l_idxes, l_idxes_length);
            float entropy_right = getEntropy(train_data.y_data_, r_idxes, r_idxes_length);

            return entropy_all - static_cast<float>(l_idxes_length)/all_idxes_vec_length * entropy_left
                               - static_cast<float>(r_idxes_length)/all_idxes_vec_length * entropy_right;
        }

        Node()
        {
        }

    public:
        Node(RandomTree* tree, const uint depth) : tree_(tree), node_type_(LEAF), depth_(depth), l_(NULL), r_(NULL), leaf_dist_(NULL), split_(0, 0, 0)
        {

        }

        ~Node()
        {
            // Recursive cleaup
            delete l_; l_ = NULL;
            delete r_; r_ = NULL;
            FREE_AND_NULL(leaf_dist_);
        }

        void train(const TrainData& train_data, const uint* subset_indices, const uint& subset_length)
        {
            bool is_leaf = (depth_ >= tree_->forest_->params_.max_depth_);

            if(tree_->forest_->params_.termcrit_type_ & RF_TERM_CRIT_NSAMPLES)
                is_leaf |= (subset_length <= tree_->forest_->params_.min_sample_cnt_);

            // check if there are different labels in the given subset. if not, this node will be a leaf
            if(!is_leaf)
            {
                YDataType val = train_data.y_data_[subset_indices[0]];
                uint i;
                for(i = 1; i < subset_length; ++i)
                {
                    if(val != train_data.y_data_[subset_indices[i]])
                        break;
                }
                if(i == subset_length)
                    is_leaf = true;
            }

            if(!is_leaf)
            {
                // Recursive child creation:
                node_type_ = SPLIT;
                uint* l_subset = NULL;
                uint l_subset_length = 0;
                uint* r_subset = NULL;
                uint r_subset_length = 0;
                split(train_data, subset_indices, subset_length, l_subset, l_subset_length, r_subset, r_subset_length);
                if(l_subset_length > 0 && r_subset_length > 0)
                {
                    l_ = new Node(tree_, depth_+1);
                    r_ = new Node(tree_, depth_+1);
                    l_->train(train_data, l_subset, l_subset_length);
                    r_->train(train_data, r_subset, r_subset_length);
                }
                else // if no split was found to separate the data, make also a leaf
                {
                    is_leaf = true;
                }
                FREE_AND_NULL(l_subset);
                FREE_AND_NULL(r_subset);
            }

            if(is_leaf)
            {
                node_type_ = LEAF;
                if(tree_->forest_->params_.store_real_probs_in_lnodes_)
                {
                    // store real probabilities in nodes (no quantitative information)
                    leaf_dist_ = calcWeightedClassProbabilitiesForYData(train_data.y_data_, subset_indices, subset_length);
                }
                else
                {
                    // store weighted sample histogram in nodes
                    leaf_dist_ = calcWeightedClassHistogramForYData(train_data.y_data_, subset_indices, subset_length);
                }

            }

        }

        // finds the best split for a given data subset (defined by direct indices convention)
        void split(const TrainData& train_data, const uint* subset_indices, const uint& subset_length,
                   uint* &l_subset, uint& l_subset_length, uint* &r_subset, uint& r_subset_length)
        {
            // Bagging in Node: only choose some samples to evaluate split (with replacement),
            // but params_.min_sample_cnt_ is still the MINIMUM!
            const uint n_bagging_samples = RF_NSAMPLES_NODE_SPLIT_BAGGING_FUNC(subset_length);
            uint* const bag_indices = Malloc(uint, n_bagging_samples);
            for(uint i = 0; i < n_bagging_samples; ++i)
            {
                const uint rand_idx = randRange<uint>(0, subset_length-1);
                bag_indices[i] = subset_indices[rand_idx];
            }

            // how many variables are considered for splitting:
            const int nvars_per_node = (int)(RF_NCHOSEN_VARS_PER_SPLIT_NODE_FUNC(train_data.f_length_));

            // choose a random subset of VARIABLES for each new node (without replacement, of course)
            uint* const chosen_var_idxes = Malloc(uint, nvars_per_node);
            bool* const chosen_var_mask = Malloc(bool, train_data.f_length_);
            memset(chosen_var_mask, 0, sizeof(bool)*train_data.f_length_);
            int cvar_cnt = 0; // chosen variable counter
            while(cvar_cnt < nvars_per_node)
            {
                int random_var_idx = randRange<int>(0,train_data.f_length_-1);
                if(!chosen_var_mask[random_var_idx])
                {
                    chosen_var_idxes[cvar_cnt] = random_var_idx;
                    chosen_var_mask[random_var_idx] = true;
                    cvar_cnt++;
                }
            }
            free(chosen_var_mask);

            // get best split of sample subset by checking thresholds of the chosen variables
            for(cvar_cnt = 0; cvar_cnt < nvars_per_node; ++cvar_cnt)
            {
                uint cur_var_idx = chosen_var_idxes[cvar_cnt];
                // get min and max value of all considered samples of the current variable
                uint cur_sample_idx = bag_indices[0];
                XDataType overall_max = train_data.x_data_[cur_sample_idx][cur_var_idx], overall_min = train_data.x_data_[cur_sample_idx][cur_var_idx];
                for(uint direct_smpl_idx_cnt = 1; direct_smpl_idx_cnt < n_bagging_samples; ++direct_smpl_idx_cnt)
                {
                    cur_sample_idx = bag_indices[direct_smpl_idx_cnt];
                    if(overall_max < train_data.x_data_[cur_sample_idx][cur_var_idx])
                    {
                        overall_max = train_data.x_data_[cur_sample_idx][cur_var_idx];
                    }
                    else if(overall_min > train_data.x_data_[cur_sample_idx][cur_var_idx])
                    {
                        overall_min = train_data.x_data_[cur_sample_idx][cur_var_idx];
                    }
                }

                // get best split score and threshold for current variable
                for(uint th_cnt = 0; th_cnt < RF_NTRIED_THRESHOLDS_PER_SPLIT_VARIABLE; ++th_cnt)
                {
                    // choose a random threshold
                    XDataType rand_th = randRange<XDataType>(overall_min, overall_max);

                    // Do and test the split

                    // allocate enough space for possible Left and Right set
                    uint* const l_bag_idxes = Malloc(uint, n_bagging_samples);
                    uint* const r_bag_idxes = Malloc(uint, n_bagging_samples);
                    uint l_bag_idxes_length = 0;
                    uint r_bag_idxes_length = 0;

                    for(uint direct_smpl_idx_cnt = 0; direct_smpl_idx_cnt < n_bagging_samples; ++direct_smpl_idx_cnt)
                    {
                        cur_sample_idx = bag_indices[direct_smpl_idx_cnt];
                        if(train_data.x_data_[cur_sample_idx][cur_var_idx] < rand_th)
                            l_bag_idxes[l_bag_idxes_length++] = cur_sample_idx;
                        else
                            r_bag_idxes[r_bag_idxes_length++] = cur_sample_idx;
                    }
                    //l_idxes = (uint*)realloc(l_idxes, sizeof(uint)*l_idxes_length); // if l_idxes_length is 0, the buffer is freed and l_idxes set to NULL!
                    //r_idxes = (uint*)realloc(r_idxes, sizeof(uint)*r_idxes_length); // same as above

                    // Calculate the score of the split (use the length to crop r and l vector, so no reallocation is needed)
                    float cur_score = informationGainForSplit(train_data, bag_indices, n_bagging_samples,
                                                           l_bag_idxes, l_bag_idxes_length,
                                                           r_bag_idxes, r_bag_idxes_length);

//                    if(cur_score <= 0)
//                    {
//                        cout << "Score is: " << cur_score << " overall max:" << overall_max << " min:" << overall_min << endl;
//                        cout << "L-length:" << l_idxes_length << " R-length:" << r_idxes_length << endl;
//                    }
                    // save the best split to Node member
                    if(cur_score > split_.score_)
                    {
                        split_.score_ = cur_score;
                        split_.var_idx_ = cur_var_idx;
                        split_.var_threshold_ = rand_th;
                    }

                    free(l_bag_idxes);
                    free(r_bag_idxes);

                }
            }

            // cleanup
            free(bag_indices);
            free(chosen_var_idxes);

            // Now do the split with all samples
            if(split_.score_ > 0)
            {
                // allocate enough space for possible Left and Right set
                l_subset = Malloc(uint, subset_length);
                r_subset = Malloc(uint, subset_length);
                l_subset_length = 0;
                r_subset_length = 0;

                for(uint sample_idx_cnt = 0; sample_idx_cnt < subset_length; ++sample_idx_cnt)
                {
                    const uint cur_sample_idx = subset_indices[sample_idx_cnt];
                    if(train_data.x_data_[cur_sample_idx][split_.var_idx_] < split_.var_threshold_)
                        l_subset[l_subset_length++] = cur_sample_idx;
                    else
                        r_subset[r_subset_length++] = cur_sample_idx;
                }
                // crop the subset
                l_subset = (uint*)realloc(l_subset, sizeof(uint)*l_subset_length);
                r_subset = (uint*)realloc(r_subset, sizeof(uint)*r_subset_length);
            }
            else
            {
                l_subset = NULL;
                l_subset_length = 0;
                r_subset = NULL;
                r_subset_length = 0;
            }


        }

        bool isLeaf() const
        {
            return node_type_ == LEAF;
        }

        const Node* const getChildForSample(const XDataType* const sample) const
        {
            if(node_type_ == LEAF)
                return NULL;

            if(sample[split_.var_idx_] < split_.var_threshold_)
                return l_;
            else
                return r_;
        }

        const float* const getLeafHist() const
        {
            return leaf_dist_;
        }

        void print() const
        {
            for(uint i = 0; i < depth_; ++i)
                cout << "-";
            cout << "Node: ";
            if(node_type_ == SPLIT)
            {
                cout << "Split: score:" << split_.score_ << " var:" << split_.var_idx_ << " th:" << split_.var_threshold_;
            }
            else if(node_type_ == LEAF)
            {
                cout << "Leaf: Hist: ";
                cout << leaf_dist_[0];
                for(uint i = 1; i < tree_->forest_->nclasses_; ++i)
                    cout << "-" << leaf_dist_[i];
            }
            cout << endl;
            if(l_)
                l_->print();
            if(r_)
                r_->print();
        }

        void save(cv::FileStorage& fs) const
        {
            fs << "type" << node_type_;
            fs << "depth" << (int)depth_;

            switch(node_type_)
            {
            case SPLIT:
            {
                fs << "split" << "{";
                fs << "var-idx" << (int)split_.var_idx_;
                fs << "var-threshold" << split_.var_threshold_;
                fs << "score" << split_.score_;
                fs << "}";

                fs << "left-child" << "{";
                l_->save(fs);
                fs << "}";
                fs << "right-child" << "{";
                r_->save(fs);
                fs << "}";
                break;
            }
            case LEAF:
            {
                fs << "histogram" << "[";
                for(uint i = 0; i < tree_->forest_->nclasses_; ++i)
                {
                    fs << leaf_dist_[i];
                }
                fs << "]";
                break;
            }
            }

        }

        void load(cv::FileNode& fn)
        {
            // just to be sure
            delete l_; l_ = NULL;
            delete r_; r_ = NULL;
            FREE_AND_NULL(leaf_dist_);

            int temp_int;
            fn["type"] >> temp_int; node_type_ = static_cast<NodeType>(temp_int);
            fn["depth"] >> temp_int; depth_ = temp_int;


            switch(node_type_)
            {
            case SPLIT:
            {
                cv::FileNode split_node = fn["split"];
                int temp_int;
                split_node["var-idx"] >> temp_int; split_.var_idx_ = temp_int;
                split_node["var-threshold"] >> split_.var_threshold_;
                split_node["score"] >> split_.score_;

                l_ = new Node(tree_, depth_+1);
                r_ = new Node(tree_, depth_+1);

                cv::FileNode left_child_node;
                left_child_node = fn["left-child"];
                l_->load(left_child_node);

                cv::FileNode right_child_node;
                right_child_node = fn["right-child"];
                r_->load(right_child_node);

                break;
            }
            case LEAF:
            {
                leaf_dist_ = Malloc(float, tree_->forest_->nclasses_);
                cv::FileNode hist = fn["histogram"];
                uint hist_val_cnt = 0;
                for(cv::FileNodeIterator hn_it = hist.begin(); hn_it != hist.end(); ++hn_it)
                {
                    float val = *hn_it;
                    leaf_dist_[hist_val_cnt++] = val;
                }
                break;
            }
            }

        }
    };

    // private random tree: not for users
    class RandomTree
    {
        friend class Node;
        friend class RandomForest;

    private:
        RandomForest* forest_;
        uint tree_id_;

        Node* root_node_;

        // compensation weights for unbalanced training data
        float* oob_error_comp_weights_;

        RandomTree()
        {

        }

    public:
        RandomTree(RandomForest* forest, uint id) : forest_(forest), tree_id_(id), oob_error_comp_weights_(NULL)
        {
            root_node_ = NULL;
        }
        ~RandomTree()
        {
            // destructor of Node also deletes its children. A recursive process
            if(root_node_)
            {
                delete root_node_;
            }
            root_node_ = NULL;

            FREE_AND_NULL(oob_error_comp_weights_);
        }

        // set_definition_vector: vector of length of the training samples, its value on a certain position specifies
        // how often a certain sample is considered for training (bootstrapping method)
        void train(const TrainData& train_data, const uint* const sample_occurance_vector, const uint& occurance_vector_length)
        {
            // destructor of Node also deletes its children. A recursive process
            if(root_node_)
                delete root_node_;

            // switch from sample occurance subset definition to direct indices
            uint* direct_indices = NULL;
            uint direct_indices_length = 0;
            sampleOccurangeToDirectSubsetIndices(sample_occurance_vector, occurance_vector_length, direct_indices, direct_indices_length);

            // Node: train method also splits and creates new nodes. Also a recursive process and the tree is finished
            root_node_ = new Node(this, 0);
            root_node_->train(train_data, direct_indices, direct_indices_length);

            FREE_AND_NULL(direct_indices);
        }

        const float* const predict(const XDataType* const sample) const
        {
            const Node* cur_node = root_node_;
            while(!cur_node->isLeaf())
            {
                cur_node = cur_node->getChildForSample(sample);
            }
            return cur_node->getLeafHist();
        }

        void print() const
        {
            cout << endl << endl << "Tree " << tree_id_ << endl << "--- BEGIN ----------------------" << endl;
            cout << "OOB-Error-comp: ";
            for(uint i = 0; i < forest_->nclasses_; ++i)
                cout << oob_error_comp_weights_[i] << ", ";

            cout << endl << "Nodes:" << endl;
            root_node_->print();
            cout << endl << "--- END ---------------------" << endl;
        }

        void save(cv::FileStorage& fs) const
        {
            fs << "id" << (int)tree_id_;
            fs << "oob-error-comp" << "[:";
            for(uint j = 0; j < forest_->nclasses_; ++j)
            {
                fs << oob_error_comp_weights_[j];
            }
            fs << "]";
            fs << "Nodes" << "{";
            root_node_->save(fs);
            fs << "}";
        }

        void load(cv::FileNode& fn)
        {
            // Destructor also deletes child nodes of root node
            delete root_node_;
            root_node_ = new Node(this, 0);

            // Tree ID
            int temp_int;
            fn["id"] >> temp_int; tree_id_ = temp_int;

            // OOB error comp. weights
            FREE_AND_NULL(oob_error_comp_weights_);
            oob_error_comp_weights_ = Malloc(float, forest_->nclasses_);
            cv::FileNode oob_err_comp_node = fn["oob-error-comp"];
            uint j = 0;
            for(cv::FileNodeIterator cw_it = oob_err_comp_node.begin(); cw_it != oob_err_comp_node.end(); ++cw_it)
            {
                oob_error_comp_weights_[j++] = *cw_it;
            }

            // All Nodes
            cv::FileNode nodes_node = fn["Nodes"];
            // load method also creates child nodes
            root_node_->load(nodes_node);
        }
    };


private:
    RandomTree** trees_;
    const Params params_;
    uint nclasses_;
    uint ntrees_;
    bool is_trained_;
    YDataType* sorted_class_labels_;
    float* class_weights_;
    float train_error_;

    void clear()
    {
        for(uint i = 0; i < ntrees_; ++i)
        {
            delete trees_[i];
        }
        FREE_AND_NULL(trees_);
        FREE_AND_NULL(sorted_class_labels_);
        FREE_AND_NULL(class_weights_);
        is_trained_ = false;
    }

    RandomForest();

public:
    RandomForest(Params params): params_(params), nclasses_(0), is_trained_(false), sorted_class_labels_(NULL), class_weights_(NULL), train_error_(0)
    {
        ntrees_ = params_.num_of_trees_in_the_forest_;
        // create the empty trees
        trees_ = Malloc(RandomTree*, ntrees_);
        for(uint i = 0; i < ntrees_; ++i)
        {
            trees_[i] = new RandomTree(this, i);
        }
    }
    ~RandomForest()
    {
        clear();
    }

    // mask_vector: Boolean array of nsamples_ length: Training samples are considered, when true is set on its position. If the Pointer is NULL, all samples are considered
    void train(const TrainData& train_data, bool* mask_vector = NULL)
    {
        if(is_trained_)
        {
            clear();
            // create empty trees
            trees_ = Malloc(RandomTree*, ntrees_);
            for(uint i = 0; i < ntrees_; ++i)
            {
                trees_[i] = new RandomTree(this, i);
            }
        }

        srand(time(NULL));

        // extract number of classes and label mapping from training data
        uniqueSortedElements<YDataType>(train_data.y_data_, train_data.nsamples_, sorted_class_labels_, nclasses_);

        // calc histogram of training data/labels
        uint* const training_data_hist = Malloc(uint, nclasses_);
        memset(training_data_hist, 0, sizeof(uint)*nclasses_);
        for(uint smp_cnt = 0; smp_cnt < train_data.nsamples_; ++smp_cnt)
        {
            // only consider all valid samples for the training data histogram
            if(!mask_vector || mask_vector[smp_cnt])
            {
                for(uint j = 0; j < nclasses_; ++j)
                {
                    if(train_data.y_data_[smp_cnt] == sorted_class_labels_[j])
                    {
                        ++training_data_hist[j];
                        break;
                    }
                }
            }
        }

        // calc weights from label histogram
        class_weights_ = Malloc(float, nclasses_);
        float class_weights_sum = 0.0f;
        for(uint j = 0; j < nclasses_; ++j)
        {
            // without sample weighting
            //class_weights_[j] = 1.0f;
            // w prop. 1-p
            //class_weights_[j] = 1.0f-static_cast<float>(training_data_hist[j])/train_data.nsamples_;
            // w prop. 1/p
            class_weights_[j] = static_cast<float>(train_data.nsamples_)/training_data_hist[j];
            class_weights_sum += class_weights_[j];
        }
        for(uint j = 0; j < nclasses_; ++j)
        {
            class_weights_[j] = class_weights_[j] / class_weights_sum;
        }
        free(training_data_hist);

        uint** oob_predict_hists = NULL;
        if(params_.calc_train_error_)
        {
            oob_predict_hists = Malloc(uint*, train_data.nsamples_);
            memset(oob_predict_hists, 0, sizeof(uint*)*train_data.nsamples_);
        }

        // train each tree
#ifndef _DEBUG
#pragma omp parallel for
#endif
        for(uint tree_cnt = 0; tree_cnt < ntrees_; ++tree_cnt)
        {
#ifdef RF_VERBOSE
            cout << "training tree nr " << tree_cnt << endl;
#endif
            // Bootstrapping by set_definition_vector: choose nsamples randomly WITH replacement:
            uint* const selection_vector = Malloc(uint, train_data.nsamples_);
            memset(selection_vector, 0, sizeof(uint)*train_data.nsamples_);
            bool* const not_selected_mask = Malloc(bool, train_data.nsamples_);
            memset(not_selected_mask, true, sizeof(bool)*train_data.nsamples_);
            for(uint i = 0; i < train_data.nsamples_; ++i)
            {
                // increment considering vector on a random position
                int chosen_sample_idx = randRange<int>(0,train_data.nsamples_-1);
                if(!mask_vector) // if mask_vector is NULL, use the sample
                    ++selection_vector[chosen_sample_idx];
                else if(mask_vector[chosen_sample_idx]) // if mask_vector is given by the user, increment only if mask-vector is not 0 on this position
                    ++selection_vector[chosen_sample_idx];

                // if samples are masked out or selected for training, in both cases
                // don't use them for weight and var-importance calculation
                not_selected_mask[chosen_sample_idx] = false;
            }
            trees_[tree_cnt]->train(train_data, selection_vector, train_data.nsamples_);

#ifdef RF_VERBOSE
            uint sum_oob_tree = 0;
            for(uint i = 0; i < train_data.nsamples_; ++i)
                sum_oob_tree += not_selected_mask[i];
            cout << sum_oob_tree << " Out-of-bag samples of " << train_data.nsamples_ << " are left for weight estimation.." << endl;
#endif
            // with not selected samples (also known as out-of-bag data) test the tree
            // to find out the weights to compensate unbalanced training data:
            // Compare the ground truth of oob-data labels to the naive (majority vote)
            // predicted labels and make them similar.
            uint* const ground_truth_hist = Malloc(uint, nclasses_);
            memset(ground_truth_hist, 0, sizeof(uint)*nclasses_);
            uint* const acc_prediction_hist = Malloc(uint, nclasses_);
            memset(acc_prediction_hist, 0, sizeof(uint)*nclasses_);
            for(uint smp_cnt = 0; smp_cnt < train_data.nsamples_; ++smp_cnt)
            {
                if(not_selected_mask[smp_cnt]) // for each oob-sample
                {
                    const XDataType* const oob_sample_x = train_data.x_data_[smp_cnt];
                    const float* const predicted_hist = trees_[tree_cnt]->predict(oob_sample_x);

                    // accumulate oob-data ground thruth and predicted histograms
                    const YDataType oob_sample_y = train_data.y_data_[smp_cnt];
                    for(uint j = 0; j < nclasses_; ++j)
                    {
                        if(oob_sample_y == sorted_class_labels_[j])
                        {
                            ++ground_truth_hist[j];
                            break;
                        }
                    }

                    // preditcion result is the leaf histogram majority vote:
                    float max = 0;
                    uint max_loc = 0;

                    maxLoc<float>(predicted_hist, nclasses_, max, max_loc);
                    ++acc_prediction_hist[max_loc];

                    if(params_.calc_train_error_)
                    {
                        // add prediction result to the result histogram for the sample over all trees
                        if(!oob_predict_hists[smp_cnt])
                        {
                            oob_predict_hists[smp_cnt] = Malloc(uint, nclasses_);
                            memset(oob_predict_hists[smp_cnt], 0, sizeof(uint)*nclasses_);
                        }
                        ++oob_predict_hists[smp_cnt][max_loc];
                    }
                }
            }

            // accumulated predition hist should be similar distributed to ground truth hist
            // calculate weights to make the acc_naive_prediction_hist and ground_truth_hist similar distributed!!
            // and store them in the tree.
            FREE_AND_NULL(trees_[tree_cnt]->oob_error_comp_weights_);
            trees_[tree_cnt]->oob_error_comp_weights_ = Malloc(float, nclasses_);

#ifdef RF_VERBOSE
            cout << "Tree oob-compensation weights are: ";
#endif

            for(uint j = 0; j < nclasses_; ++j)
            {
                if(acc_prediction_hist[j]>0)
                    trees_[tree_cnt]->oob_error_comp_weights_[j] = (float)ground_truth_hist[j]/acc_prediction_hist[j]; // change to 1, if no OOB compensation should be performed
                else
                    trees_[tree_cnt]->oob_error_comp_weights_[j] = 1;

#ifdef RF_VERBOSE
                cout << trees_[tree_cnt]->oob_error_comp_weights_[j] << ", ";
#endif
            }

#ifdef RF_VERBOSE
            cout << endl;
#endif
            free(ground_truth_hist);
            free(acc_prediction_hist);
            free(selection_vector);
            free(not_selected_mask);
        } // for each tree

        if(params_.calc_train_error_)
        {
            // calc prediction over all trees
            uint sum_false = 0;
            uint sum_ever_oob = 0; // number of samples that ever have been oob
            for(uint smp_cnt = 0; smp_cnt < train_data.nsamples_; ++smp_cnt)
            {
                if(oob_predict_hists[smp_cnt])
                {
                    uint max = 0, max_loc = 0;
                    maxLoc<uint>(oob_predict_hists[smp_cnt], nclasses_, max, max_loc);

                    if(sorted_class_labels_[max_loc] != train_data.y_data_[smp_cnt])
                    {
                        ++sum_false;
                    }

                    ++sum_ever_oob;
                }
                FREE_AND_NULL(oob_predict_hists[smp_cnt]);
            }
            FREE_AND_NULL(oob_predict_hists);

            train_error_ = static_cast<float>(sum_false)/sum_ever_oob;
        }

        is_trained_ = true;

#ifdef RF_VERBOSE_PRINT_TRAIN_ERROR
        cout << "RandomForest training finished. #classes: " << nclasses_ << ", train error: " << train_error_ << endl;
#endif
//#ifdef RF_VERBOSE
//        print();
//#endif

    }


    void predict(const XDataType* const sample, YDataType* &result, float* &prop, uint& prop_length, const PredictReturnType &ptype) const
    {
        assert(is_trained_);

        if(ptype == SINGLE_CLASS_LABEL || ptype == SINGLE_CLASS_LABEL_PROP || ptype == CUMULATIVE_HIST)
        {
            float* const cum_hist = Malloc(float, nclasses_);
            memset(cum_hist, 0.0f, sizeof(float)*nclasses_);
            float cum_hist_sum = 0.0;
//#ifndef _DEBUG
//#pragma omp parallel for // is faster without omp
//#endif
            for(uint i = 0; i < ntrees_; ++i)
            {
                const float* const node_hist = trees_[i]->predict(sample);
                const float* const comp_weights = trees_[i]->oob_error_comp_weights_;

                // majority vote + unbalanced class compensation
                float max = 0;
                uint max_loc = 0;
                maxLoc<float>(node_hist, nclasses_, max, max_loc);
                cum_hist[max_loc] += comp_weights[max_loc];
                cum_hist_sum += comp_weights[max_loc];

//                other variant: HISTOGRAM ADDING
//                for(uint j = 0; j < nclasses_; ++j)
//                {
//                    cum_hist[j] += static_cast<float>(node_hist[j])*comp_weights[j];
//                    cum_hist_sum += cum_hist[j];
//                }
            }

            if(ptype == SINGLE_CLASS_LABEL)
            {
                float max = 0;
                uint max_loc = 0;
                maxLoc<float>(cum_hist, nclasses_, max, max_loc);
                result = Malloc(YDataType, 1);
                *result = sorted_class_labels_[max_loc];

                free(cum_hist);
                prop = NULL;
                prop_length = 0;
                return;
            }
            else if(ptype == SINGLE_CLASS_LABEL_PROP)
            {
                float max = 0;
                uint max_loc = 0;
                maxLoc<float>(cum_hist, nclasses_, max, max_loc);
                result = Malloc(YDataType, 1);
                *result = sorted_class_labels_[max_loc];

                free(cum_hist);
                prop = Malloc(float,1);
                *prop = max/cum_hist_sum;
                prop_length = 1;
                return;
            }
            else if(ptype == CUMULATIVE_HIST)
            {
                result = Malloc(YDataType, nclasses_);
                for(uint j = 0; j < nclasses_; ++j)
                {
                    result[j] = sorted_class_labels_[j];
                    cum_hist[j] = cum_hist[j]/cum_hist_sum;
                }
                prop = cum_hist;
                prop_length = nclasses_;
                return;
            }

        }
        else if(ptype == TREE_HISTS)
        {
            float* const leaf_hists = Malloc(float, ntrees_*nclasses_);
//#ifndef _DEBUG
//#pragma omp parallel for // is faster without omp
//#endif
            for(uint i = 0; i < ntrees_; ++i)
            {
                const float* const node_hist = trees_[i]->predict(sample);
                const float* const comp_weights = trees_[i]->oob_error_comp_weights_;
                for(uint j = 0; j < nclasses_; ++j)
                {
                    leaf_hists[i*nclasses_ + j] = static_cast<float>(node_hist[j])*comp_weights[j];
                }
            }
            prop = leaf_hists;
            prop_length = ntrees_*nclasses_;
        }
    }

    const uint getNumberOfClasses() const
    {
        return nclasses_;
    }

    void print() const
    {
        cout << "Printing whole Random Forest: " << endl;
        for(uint i = 0; i < ntrees_; ++i)
        {
            trees_[i]->print();
        }
    }

    void save(cv::FileStorage& fs, const char* xml_tag) const
    {
        fs << xml_tag << "{";

        fs << "nclasses" << (int)nclasses_;
        fs << "ntrees" << (int)ntrees_;
        fs << "sorted-labels" << "[:";
        for(uint i = 0; i < nclasses_; ++i)
            fs << sorted_class_labels_[i];
        fs << "]";
        fs << "class-weights" << "[:";
        for(uint i = 0; i < nclasses_; ++i)
            fs << class_weights_[i];
        fs << "]";

        fs << "training-error" << train_error_;

        fs << "Trees" << "[";
        for(uint i = 0; i < ntrees_; ++i)
        {
            fs << "{";
            trees_[i]->save(fs);
            fs << "}";
        }
        fs << "]";

        fs << "}";
    }

    void load(cv::FileNode& rf_node)
    {
        clear();
        int int_temp;
        rf_node["nclasses"] >> int_temp;
        nclasses_ = int_temp;
        rf_node["ntrees"] >> int_temp;
        ntrees_ = int_temp;

        sorted_class_labels_ = Malloc(YDataType, nclasses_);
        cv::FileNode scl_node = rf_node["sorted-labels"];
        uint scl_cnt = 0;
        for(cv::FileNodeIterator it = scl_node.begin(); it != scl_node.end(); ++it)
        {
            *it >> sorted_class_labels_[scl_cnt++];
        }

        class_weights_ = Malloc(float, nclasses_);
        cv::FileNode cw_node = rf_node["class-weights"];
        uint cw_cnt = 0;
        for(cv::FileNodeIterator it = cw_node.begin(); it != cw_node.end(); ++it)
        {
            *it >> class_weights_[cw_cnt++];
        }

        rf_node["training-error"] >> train_error_;

        trees_ = Malloc(RandomTree*, ntrees_);
        for(uint i = 0; i < ntrees_; ++i)
        {
            trees_[i] = new RandomTree(this, i);
        }
        cv::FileNode trees_node = rf_node["Trees"];
        uint tree_cnt = 0;
        for(cv::FileNodeIterator it = trees_node.begin(); it != trees_node.end(); ++it)
        {
            cv::FileNode tn = *it;
            trees_[tree_cnt++]->load(tn);
        }

        is_trained_ = true;
    }
};


}

#endif // RANDOMFOREST_H
