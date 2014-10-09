#ifndef VISIONMODULE_H_
#define VISIONMODULE_H_

#include "VisionData.h"
#include "Mat.h"
#include "Vector.h"
#include "Exceptions.h"

#include <string>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>

#define UNKNOWN_DATA_LENGTH    -1

#define MODULE_CONSTRUCTOR_SIGNATURE  bool is_root, string module_params

#define MODULE_CTOR_INIT(print_name, required_sig, output_sig) \
	module_print_name_ = print_name; \
	required_input_signature_ = required_sig; \
	output_signature_ = output_sig; \
	if(is_root)	setAsRoot();

using namespace std;
using namespace cv;

struct NormVals
{
	float mean_;
	float stddev_;
};


class CVisionModule
{
private:
	void addAncestor(CVisionModule* anchestor); // ONLY called by setSuccessor from an other CVisionModule Instance

protected:
	CVisionModule();

	void setAsRoot();
	CVisionData getConcatenatedDataAndClearBuffer();
//	int module_type_;
	string module_print_name_;
	string module_id_;
	bool is_trained_;

	int required_input_signature_;
	int output_signature_;

	// coherent vectors (one element per anchestor)
	vector<CVisionModule*> ancestors_;
	vector<CVisionDataConverter*> data_converters_;
	vector<CVisionData*> data_buffers_;
	//vector<NormVals> data_norm_vals_;

	CVisionModule* successor_;

	CVisionData* data_labels_;

public:
	virtual ~CVisionModule();

	void setModuleID(const int& module_id);

	string getModuleID() const;

	void setSuccessor(CVisionModule*);

	inline bool isTrained() const
	{
		return is_trained_;
	}

	inline void setAsTrained()
	{
		is_trained_ = true;
	}

	inline CVisionModule* getSuccessor()
	{
		return successor_;
	}

	bool isRoot() const;

	CVisionModule* getAncestorModuleFromWhichNoDataIsBuffered();

	inline int getOutputSignature() const
	{
		return output_signature_;
	}

	inline string getPrintName() const
	{
		return module_print_name_;
	}

	// converts the data and fills the data buffer in order to train or execute the modules functionality
	void bufferData(const CVisionData* data, const CVisionModule* from_module);
	void setDataLabels(const CVisionData& labels);
	void clearDataBuffers();

	// VIRTUAL Methods
	// executes the modules functionality with the buffered data
	virtual CVisionData* exec() = 0;

	// trains the module according to the buffered data and labels
	virtual void train();

	virtual void save(FileStorage& fs) const = 0;

	virtual void load(FileStorage& fs) = 0;
};


/*
 * Class CVisionDataMergingModule:
 * Takes VisionData from different channels and merges it to one VECTOR
 */
//class CVisionDataMergingModule
//{
//private:
//	CVisionDataMergingModule();
//	vector<CVisionDataConverter*> channel_converters_;
//
//public:
//	CVisionDataMergingModule(const vector<int>& channel_end_signatures)
//    {
//		for(vector<int>::const_iterator sig_it = channel_end_signatures.begin(); sig_it != channel_end_signatures.end(); ++sig_it)
//		{
//			CVisionDataConverter* cur_converter = NULL;
//			// if data that comes from a channel is not ready for merging i.e. either a float vector of a float scalar
//			if(!SIGNATURE_IS_FLOAT(*sig_it) || !(SIGNATURE_IS_VECTOR(*sig_it) || SIGNATURE_IS_SCALAR(*sig_it)))
//			{
//				cur_converter = new CVisionDataConverter(*sig_it, DATA_TYPE_VECTOR | CV_32FC1);
//			}
//		}
//	}
//
//	void merge(const vector<CVisionData>& inputs, CVisionData& output)
//	{
//
//	}
//};

#endif
