#include "VisionModule.h"


class CPreprocessingModule : public CVisionModule
{
public:
	CPreprocessingModule();
	virtual ~CPreprocessingModule();

//	virtual void exec(std::vector<CVisionData*>& data) throw(VisionDataTypeException) = 0;
};

