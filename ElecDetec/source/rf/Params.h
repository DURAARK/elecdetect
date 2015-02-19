/*
 * RandomForest Implementation
 * Params.h
 *
 *  Created on: Okt, 2014
 *      Author: Robert Viehauser
 *              robert(dot)viehauser(at)gmail(dot)com
 */


#ifndef RFPARAMS_H
#define RFPARAMS_H

#include "Utils.h"

#define RF_TERM_CRIT_NSAMPLES    1

namespace RF
{

struct Params
{
    uint max_depth_;  // maximum depth of a tree in the forest
    uint num_of_trees_in_the_forest_;  // number of trees in the forest
    int termcrit_type_;    // specifies when a node is not split anymore (aside the trees depth)
    uint min_sample_cnt_;  // number of samples at which no splitting is performed (only works with termcrit_type_ & RF_TERM_CRIT_NSAMPES)
    bool calc_train_error_;  // calculate the training set error by using the out-of-bag data
    bool store_real_probs_in_lnodes_; // if true, the leaf node store a true probability distribution in the leaf nodes.
                                      // Otherwise, the (weighted: determined by the training data class distribution) amount of samples are stored as histogram

    Params()
    {
        max_depth_ = 15;
        num_of_trees_in_the_forest_ = 50;
        termcrit_type_ = RF_TERM_CRIT_NSAMPLES;
        min_sample_cnt_ = 10;
        calc_train_error_ = false;
        store_real_probs_in_lnodes_ = false;
    }

    Params(const uint& max_depth, const uint& max_num_of_trees_in_the_forest, const int& termcrit_type, const uint& min_sample_cnt, const bool& calc_train_error, const bool& store_real_probs_in_lnodes)
    {
        max_depth_ = max_depth < 1 ? 1 : max_depth;
        num_of_trees_in_the_forest_ = max_num_of_trees_in_the_forest < 1 ? 1 : max_num_of_trees_in_the_forest;
        termcrit_type_ = termcrit_type;
        min_sample_cnt_ = min_sample_cnt < 1 ? 1 : min_sample_cnt;
        calc_train_error_ = calc_train_error;
        store_real_probs_in_lnodes_ = store_real_probs_in_lnodes;
    }

};

}


#endif // RFPARAMS_H
