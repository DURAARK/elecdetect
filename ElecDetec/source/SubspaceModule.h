/*
 * SubspaceModule.h
 *
 *  Created on: Jul 31, 2014
 *      Author: test
 */

#ifndef SUBSPACEMODULE_H_
#define SUBSPACEMODULE_H_

#include "VisionModule.h"
#include "Scalar.h"
#include "Vector.h"
#include "VectorArray.h"
#include "Mat.h"

class CSubspaceModule: public CVisionModule
{
public:
	CSubspaceModule();
	virtual ~CSubspaceModule();

	virtual void exec(std::vector<CVisionData*>& data) throw(VisionDataTypeException) = 0;

	// train_data contains for each sample one CVisionData and train_labels a CVisionData-Label
	virtual void train(const CMat& train_data, const CVector<int>& train_labels) = 0;

	virtual void save(FileStorage& fs) const = 0;

	virtual void load(FileStorage& fs) = 0;
};

#endif /* SUBSPACEMODULE_H_ */
