#include "VisionModule.h"


CVisionModule::CVisionModule() : module_type_(MOD_TYPE_UNKNOWN), needs_training_(false)
{
}


CVisionModule::~CVisionModule()
{
}

void CVisionModule::train(const CMat& train_data, const CVector<int>& train_labels)
{
	cout << "nothing to train here... have you made an mistake?" << endl;
}

void CVisionModule::setModuleID(const int& module_id)
{
	stringstream ss_mod_id;
	ss_mod_id << "module-" << module_id;
	module_id_ = ss_mod_id.str();
}
