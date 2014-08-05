#ifndef VISIONMODULE_H_
#define VISIONMODULE_H_

#include "VisionData.h"
#include "Exceptions.h"

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>


#define  MOD_TYPE_PREPROC       1 << 0
#define  MOD_TYPE_FEATURE       1 << 1
#define  MOD_TYPE_SUBSPACE      1 << 2
#define  MOD_TYPE_CLASSIFIER    1 << 3

using namespace std;
using namespace cv;

class CVisionModule
{
protected:
	int module_type_;
	string module_name_;

public:
	CVisionModule();
	virtual ~CVisionModule();

	virtual void exec(vector<CVisionData*>& data) throw(VisionDataTypeException) = 0;

	inline int getType() {
		return module_type_;
	}

	inline string getName() {
		return module_name_;
	}

	virtual void save(FileStorage& fs) const = 0;

	virtual void load(FileStorage& fs) = 0;
};

#endif
