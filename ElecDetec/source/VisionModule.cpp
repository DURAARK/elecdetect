#include "VisionModule.h"


CVisionModule::CVisionModule() : needs_training_(false), output_type_(0), required_input_signature_mask_(0), data_converter_(NULL)
{
}


CVisionModule::~CVisionModule()
{
}

void CVisionModule::train(const CVisionData& train_data, const CVisionData& train_labels)
{
	cout << "nothing to train here... have you made an mistake?" << endl;
}

void CVisionModule::setModuleID(const int& module_id)
{
	stringstream ss_mod_id;
	ss_mod_id << "module-" << module_id;
	module_id_ = ss_mod_id.str();
}
