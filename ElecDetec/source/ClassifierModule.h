#ifndef CCLASSIFIERMODULE_H_
#define CCLASSIFIERMODULE_H_


#include "VisionModule.h"
#include "Scalar.h"
#include "Vector.h"
#include "VectorArray.h"
#include "Mat.h"


class CClassifierModule :
	public CVisionModule
{
public:
	CClassifierModule();
	virtual ~CClassifierModule();

	virtual void exec(std::vector<CVisionData*>& data) throw(VisionDataTypeException) = 0;

	// train_data contains for each sample one CVisionData and train_labels a CVisionData-Label
	virtual void train(const CMat& train_data, const CVector<int>& train_labels) throw(VisionDataTypeException) = 0;

	virtual void save(FileStorage& fs) const = 0;

	virtual void load(FileStorage& fs) = 0;
};

#endif
