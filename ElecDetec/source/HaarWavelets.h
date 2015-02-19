/*
 * ElecDetec: HaarWavelets.h
 *
 *  Created on: Feb, 2015
 *      Author: Robert Viehauser
 */

#ifndef HAARWAVELETS_H_
#define HAARWAVELETS_H_

#include <Defines.h>
#include <Utils.h>
#include <time.h>
#include <Debug.h>

#include <sstream>

// Haar-Like Feature Parameter
#define HAAR_FEAT_N_TESTS                        3000  // Number of testpairs
#define HAAR_FEAT_SYM_RATIO                       0.2  // Percentage of forced symmetric arranged testpairs
#define HAAR_FEAT_MIN_RECT_SZ                       4

using namespace std;
using namespace cv;

extern int _PATCH_WINDOW_SIZE_;

class CHaarWavelets
{
    struct TestRect
    {
        Rect rect_;
        float ch_indicator_;
        TestRect(Rect rect, float chind) :
        rect_(rect),
        ch_indicator_(chind)
        { }
    };

    typedef pair<TestRect, TestRect> TestPair;

private:
	int n_tests_;
    float symm_ratio_;
    float pair_ratio_;
    int min_rect_sz_;
    int max_rect_sz_;

    vector<TestPair> wavelet_pairs_;

	void initWavelets();

public:
    CHaarWavelets();
    ~CHaarWavelets();

    void extractFeatureVector(const vector<Mat>& input_channels, vector<float>& output_vec);
    void save(FileStorage& fs) const;
    void load(FileNode& fs);
};

#endif /* HAARWAVELETS_H_ */
