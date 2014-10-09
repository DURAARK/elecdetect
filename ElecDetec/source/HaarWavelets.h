/*
 * HaarWavelets.h
 *
 *  Created on: Oct 6, 2014
 *      Author: test
 */

#ifndef HAARWAVELETS_H_
#define HAARWAVELETS_H_

#include "Defines.h"
#include "Utils.h"
#include "VisionModule.h"
#include "VisionData.h"

#include <sstream>

using namespace std;
using namespace cv;



class CHaarWavelets: public CVisionModule
{
	typedef pair<Rect, Rect> RectPair;

private:
	int n_tests_;
	int symm_percentage_;

	vector<RectPair> wavelet_pairs_;

	void initWavelets();

	CHaarWavelets();
public:
	CHaarWavelets(MODULE_CONSTRUCTOR_SIGNATURE);
	virtual ~CHaarWavelets();

	virtual CVisionData* exec();
	virtual void save(FileStorage& fs) const;
	virtual void load(FileStorage& fs);
};

#endif /* HAARWAVELETS_H_ */
