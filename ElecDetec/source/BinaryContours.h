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
	CBinaryContours(int expected_input_signature);
	~CBinaryContours();

	virtual void exec(const CVisionData& input_data, CVisionData& output_data);
	virtual void save(FileStorage& fs) const;
	virtual void load(FileStorage& fs);
};

