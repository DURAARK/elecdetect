#ifndef VISIONMODULE_H_
#define VISIONMODULE_H_

#include "VisionData.h"
#include "Mat.h"
#include "Vector.h"
#include "Exceptions.h"

#include <string>
#include <strstream>
#include <vector>
#include <opencv2/opencv.hpp>

#define  MOD_TYPE_UNKNOWN       0
#define  MOD_TYPE_PREPROC       1 << 0
#define  MOD_TYPE_FEATURE       1 << 1
#define  MOD_TYPE_SUBSPACE      1 << 2
#define  MOD_TYPE_CLASSIFIER    1 << 3

#define UNKNOWN_DATA_LENGTH    -1

using namespace std;
using namespace cv;

class CVisionModule
{
protected:
	int module_type_;
	string module_print_name_;
	string module_id_;
	bool needs_training_;

public:
	CVisionModule();
	virtual ~CVisionModule();

	void setModuleID(const int& module_id);

	virtual void exec(vector<CVisionData*>& data) throw(VisionDataTypeException) = 0;

	// train_data contains for each sample one CVisionData and train_labels a CVisionData-Label
	virtual void train(const CMat& train_data, const CVector<int>& train_labels);

	inline bool needsTraining()
	{
		return needs_training_;
	}

	inline int getType()
	{
		return module_type_;
	}

	inline string getPrintName()
	{
		return module_print_name_;
	}

	virtual void save(FileStorage& fs) const = 0;

	virtual void load(FileStorage& fs) = 0;
};

#endif
