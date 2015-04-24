/*
 * ElecDetec: HaarWavelets.cpp
 *
 *  Created on: Feb, 2015
 *      Author: Robert Viehauser
 */


#include "HaarWavelets.h"

CHaarWavelets::CHaarWavelets() :
    n_tests_(HAAR_FEAT_N_TESTS),
    symm_ratio_(HAAR_FEAT_SYM_RATIO),
    min_rect_sz_(HAAR_FEAT_MIN_RECT_SZ),
    max_rect_sz_(_PATCH_WINDOW_SIZE_)
{
    srand(time(NULL));
	initWavelets();
}

CHaarWavelets::~CHaarWavelets()
{

}


void CHaarWavelets::initWavelets()
{
    // generate test-boxes (relative coordinates within the image patch)
    wavelet_pairs_.clear();
    for(int i = 0; i < n_tests_; ++i)
    {
        Rect rect1;
        rect1.width = randRange<int>(min_rect_sz_, max_rect_sz_);
        rect1.height = randRange<int>(min_rect_sz_, max_rect_sz_);
        rect1.x = randRange<int>(0, _PATCH_WINDOW_SIZE_-rect1.width-1);
        rect1.y = randRange<int>(0, _PATCH_WINDOW_SIZE_-rect1.height-1);

        Rect rect2(0,0,0,0);
        if(symm_ratio_ > randRange<float>(0,1)) // should the pairs be symmetric?
        {
            // Percentage of the types of the symmetric arranged pairs:
            // Horizontal-symmetric, Vertical-symmetric, Center-symmetric
            const float percent_h_sym = 0.3;
            const float percent_v_sym = 0.3; // the rest is center-symmetric

            float sym_type = randRange<float>(0,1);

            rect2.height = rect1.height;
            rect2.width = rect1.width;
            if(sym_type < percent_h_sym) // h-symmetric
            {
                rect2.x = _PATCH_WINDOW_SIZE_ - rect1.x - rect1.width; // see notes for derivation
                rect2.y = rect1.y;
            }
            else if(sym_type < percent_h_sym+percent_v_sym) // v-symmetric
            {
                rect2.x = rect1.x;
                rect2.y = _PATCH_WINDOW_SIZE_ - rect1.y - rect1.height;
            }
            else // c-symmetric
            {
                rect2.x = _PATCH_WINDOW_SIZE_ - rect1.x - rect1.width;
                rect2.y = _PATCH_WINDOW_SIZE_ - rect1.y - rect1.height;
            }
        }
        else
        {
            rect2.width = randRange<int>(min_rect_sz_, max_rect_sz_);
            rect2.height = randRange<int>(min_rect_sz_, max_rect_sz_);
            rect2.x = randRange<int>(0, _PATCH_WINDOW_SIZE_-rect2.width-1);
            rect2.y = randRange<int>(0, _PATCH_WINDOW_SIZE_-rect2.height-1);
        }

        float ch_indicator1 = randRange<float>(0,1);
        float ch_indicator2 = ch_indicator1; // use the same channel for the second rect. // randRange<float>(0,1);

        TestRect testrect1(rect1, ch_indicator1);
        TestRect testrect2(rect2, ch_indicator2);

        wavelet_pairs_.push_back(TestPair(testrect1, testrect2));
    }
}


void CHaarWavelets::extractFeatureVector(const vector<Mat>& input_channels, vector<float>& output_vec)
{
    output_vec.clear();
    output_vec.reserve(wavelet_pairs_.size());

    uint nchannels = input_channels.size();

    vector<Mat> integral_channels;
    integral_channels.reserve(nchannels);

    for(uint ch_cnt = 0; ch_cnt < nchannels; ++ch_cnt)
    {
        //PAUSE_AND_SHOW(input_channels[ch_cnt]);
        Mat integral_img;
        integral(input_channels[ch_cnt], integral_img, CV_32S);
        integral_channels.push_back(integral_img);
    }

    for(vector<TestPair>::const_iterator pair_it = wavelet_pairs_.begin(); pair_it != wavelet_pairs_.end(); ++pair_it)
	{
		float feature;

        const Point tl1 = pair_it->first.rect_.tl();
        const Point tr1(pair_it->first.rect_.x + pair_it->first.rect_.width, pair_it->first.rect_.y);
        const Point br1 = pair_it->first.rect_.br();
        const Point bl1(pair_it->first.rect_.x, pair_it->first.rect_.y + pair_it->first.rect_.height);

        uint channel1 = floor((pair_it->second.ch_indicator_)*nchannels);
        channel1 = channel1 >= nchannels ? nchannels-1 : channel1; // if ch_indicator is exactly 1.0

        //cout << "nchannels: " << nchannels << " channel1: " << channel1 << " , chind: " << pair_it->first.ch_indicator_ << endl;

        const int rect1_sum = integral_channels[channel1].at<int>(br1) - integral_channels[channel1].at<int>(bl1) -
                              integral_channels[channel1].at<int>(tr1) + integral_channels[channel1].at<int>(tl1);
        const float rect1_mean = (float)rect1_sum/pair_it->first.rect_.area();

        const Point tl2 = pair_it->second.rect_.tl();
        const Point tr2(pair_it->second.rect_.x + pair_it->second.rect_.width, pair_it->second.rect_.y);
        const Point br2 = pair_it->second.rect_.br();
        const Point bl2(pair_it->second.rect_.x, pair_it->second.rect_.y + pair_it->second.rect_.height);

        uint channel2 = floor((pair_it->first.ch_indicator_)*nchannels);
        channel2 = channel2 >= nchannels ? nchannels-1 : channel2; // if ch_indicator is exactly 1.0

        const int rect2_sum = integral_channels[channel2].at<int>(br2) - integral_channels[channel2].at<int>(bl2) -
                              integral_channels[channel2].at<int>(tr2) + integral_channels[channel2].at<int>(tl2);
        const float rect2_mean = (float)rect2_sum/pair_it->second.rect_.area();
        feature = (rect1_mean - rect2_mean)/255.0;

        output_vec.push_back(feature);
	}
}

void CHaarWavelets::save(FileStorage& fs) const
{
    fs << "n-tests" << n_tests_;
    fs << "symm-percent" << symm_ratio_;
    fs << "rect-pairs" << "[:";
    for(vector<TestPair>::const_iterator pair_it = wavelet_pairs_.begin(); pair_it != wavelet_pairs_.end(); ++pair_it)
    {
        fs << "{";
        fs << "Rect1" << "{";
        fs << "rect" << pair_it->first.rect_;
        fs << "chind" << pair_it->first.ch_indicator_;
        fs << "}"; // Rect1

        fs << "Rect2" << "{";
        fs << "rect" << pair_it->second.rect_;
        fs << "chind" << pair_it->second.ch_indicator_;
        fs << "}"; // Rect2
        fs << "}"; // TestPair
    }
	fs << "]";
}

void CHaarWavelets::load(FileNode& node)
{
	wavelet_pairs_.clear();

    node["n-tests"] >> n_tests_;
    node["symm-percent"] >> symm_ratio_;

    FileNode rect_pairs_node = node["rect-pairs"];
	for(FileNodeIterator pair_node_it = rect_pairs_node.begin(); pair_node_it != rect_pairs_node.end(); ++pair_node_it)
	{
        FileNode testpair_node = (*pair_node_it);
        FileNode rect1_node = testpair_node["Rect1"];
        FileNode rect2_node = testpair_node["Rect2"];

		Rect rect1, rect2;
        rect1_node["rect"] >> rect1;
        rect2_node["rect"] >> rect2;

        float chind1, chind2;
        rect1_node["chind"] >> chind1;
        rect2_node["chind"] >> chind2;

        TestRect testrect1(rect1, chind1);
        TestRect testrect2(rect2, chind2);

        //cout << "TestRect: 1: Rect: " << testrect1.rect_ << ", chind: " << testrect1.ch_indicator_ << endl;
        //cout << "TestRect: 2: Rect: " << testrect2.rect_ << ", chind: " << testrect2.ch_indicator_ << endl;

        wavelet_pairs_.push_back(TestPair(testrect1, testrect2));
	}

}
