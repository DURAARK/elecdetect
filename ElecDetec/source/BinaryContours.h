#pragma once
#include "PreprocessingModule.h"
#include "Mat.h"

#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

class CBinaryContours :
	public CPreprocessingModule
{
public:
	CBinaryContours();
	~CBinaryContours();

	virtual void exec(std::vector<CVisionData*>& data) throw(VisionDataTypeException);
	virtual void save(FileStorage& fs) const;
	virtual void load(FileStorage& fs);
};

