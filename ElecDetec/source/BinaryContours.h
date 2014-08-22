#include <opencv2/opencv.hpp>
#include <vector>

#include "VisionModule.h"
#include "VisionData.h"

using namespace std;
using namespace cv;

class CBinaryContours : public CVisionModule
{
private:
	CBinaryContours();

public:
	CBinaryContours(MODULE_CONSTRUCTOR_SIGNATURE);
	~CBinaryContours();

	virtual CVisionData* exec();
	virtual void save(FileStorage& fs) const;
	virtual void load(FileStorage& fs);
};

