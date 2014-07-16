#pragma once
#include "VisionModule.h"
class CFeatureExtractorModule :
	public CVisionModule
{
public:
	CFeatureExtractorModule();
	virtual ~CFeatureExtractorModule();

	virtual void exec(std::vector<CVisionData*>& data) throw(VisionDataTypeException) = 0;
	virtual int getFeatureLength() = 0;

};

