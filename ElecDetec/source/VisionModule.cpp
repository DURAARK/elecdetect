#include "VisionModule.h"


CVisionModule::CVisionModule() : is_trained_(true), required_input_signature_(0), output_signature_(0), successor_(NULL), data_labels_(NULL)
{
}


CVisionModule::~CVisionModule()
{
	for(vector<CVisionDataConverter*>::iterator dc_it = data_converters_.begin(); dc_it != data_converters_.end(); ++dc_it)
	{
		if(*dc_it)
			delete *dc_it;
		*dc_it = NULL;
	}

	clearDataBuffers();

	if(data_labels_)
		delete data_labels_;
	data_labels_ = NULL;
}

void CVisionModule::bufferData(const CVisionData* data, const CVisionModule* ancestor_module)
{
	// search correct converters and buffers for the ancestor from which the data comes from

	vector<CVisionModule*>::const_iterator anc_it;
	vector<CVisionDataConverter*>::iterator conv_it;
	vector<CVisionData*>::iterator buf_it;
	for(anc_it = ancestors_.begin(), buf_it = data_buffers_.begin(), conv_it = data_converters_.begin(); anc_it != ancestors_.end(); ++anc_it, ++buf_it, ++conv_it)
	{
		if(*anc_it == ancestor_module) // if the pointers have the same value
		{
			break;
		}
	}
	if(anc_it == ancestors_.end())
	{
		cerr << "CVisionModule::bufferData: Ancestor Module not found!!!" << endl;
		exit(-1);
	}

	// create new Data Instance with write permissions. this does not copy the cv::Mat content!
	CVisionData* temp_buffer = new CVisionData(data->data(), data->getType());
	// if data needs to be converted (otherwise the converter-Pointer is NULL)
	if(*conv_it)
	{
		(*conv_it)->convert(*temp_buffer); // this creates a new cv::Mat content if necessary
	} // otherwise temp_buffer->data().data == data.data().data (i.e. cv::Mat::data - Pointer)


	// if the data-buffer is empty
	if(!(*buf_it))
	{
		*buf_it = temp_buffer;
	}
	else // otherwise add samples to current buffered data for training
	{
		if(this->is_trained_)
		{
			cerr << "CVisionModule::bufferData: Buffer already contains data, but training is already finished..." << endl;
			exit(-1);
		}
		if((*buf_it)->data().cols != temp_buffer->data().cols)
		{
			cerr << "Different feature dimension when concatenating samples! Exiting..." << endl;
			exit(-1);
		}
		(*buf_it)->concatenateRowwise(*temp_buffer);
		delete temp_buffer;
	}
	temp_buffer = NULL;

}

CVisionData* CVisionModule::getConcatenatedDataAndClearBuffer()
{
	if(data_buffers_.empty())
	{
		cerr << "CVisionModule::getConcatenatedDataAndClearBuffer: no Buffer objects?" << endl;
		exit(-1);
	}
	for(vector<CVisionData*>::iterator buf_it = data_buffers_.begin(); buf_it != data_buffers_.end(); ++buf_it)
	{
		if(!(*buf_it) || (*buf_it)->data().empty())
		{
			cerr << "CVisionModule::getConcatenatedDataAndClearBuffer: at least one Buffer is empty!" << endl;
			exit(-1);
		}
	}
	CVisionData* concatenated = new CVisionData(data_buffers_.front()->data(), data_buffers_.front()->getType());
	for(vector<CVisionData*>::iterator buf_it = data_buffers_.begin()+1; buf_it != data_buffers_.end(); ++buf_it)
	{
		if(!(*buf_it) || (*buf_it)->data().empty())
		{
			cerr << "CVisionModule::getConcatenatedDataAndClearBuffer: at least one Buffer is empty!" << endl;
			exit(-1);
		}
		concatenated->concatenateColumnwise(**buf_it);
	}
	clearDataBuffers();
	return concatenated;
}


CVisionModule* CVisionModule::getAncestorModuleFromWhichNoDataIsBuffered()
{
	for(vector<CVisionData*>::iterator buf_it = data_buffers_.begin(); buf_it != data_buffers_.end(); ++buf_it)
	{
		if(!(*buf_it) || (*buf_it)->data().empty())
		{
			return ancestors_[distance(data_buffers_.begin(), buf_it)];
		}
	}
	return NULL;
}

void CVisionModule::setAsRoot()
{
	CVisionDataConverter* new_converter = NULL;
	if( (DATA_TYPE_IMAGE | CV_8UC3) != required_input_signature_)
	{
		new_converter = new CVisionDataConverter((DATA_TYPE_IMAGE | CV_8UC3), required_input_signature_);
	}
	data_converters_.push_back(new_converter);
	data_buffers_.push_back(NULL);
	if(!ancestors_.empty())
	{
		cerr << "CVisionModule: Root Module has Ancestors?!" << endl;
		exit(-1);
	}
	ancestors_.push_back(NULL); // Root Module has one NULL-ancestor
}

void CVisionModule::addAncestor(CVisionModule* ancestor)
{
	CVisionDataConverter* new_converter = NULL;

	// if datatypes do not match, create a converter for the ancestor. otherwise NULL is stored
	if(ancestor->getOutputSignature() != required_input_signature_)
	{
		new_converter = new CVisionDataConverter(ancestor->getOutputSignature(), required_input_signature_);
	}

	data_converters_.push_back(new_converter);
	data_buffers_.push_back(NULL);
	ancestors_.push_back(ancestor);
}

void CVisionModule::setSuccessor(CVisionModule* successor)
{
	if(successor_)
	{
		cerr << "Two successors? Really...?" << endl;
		exit(-1);
	}
	successor_ = successor;
	successor_->addAncestor(this); // add this module as an ancestor of the successor module
}


void CVisionModule::setDataLabels(const CVisionData& labels)
{
	if(data_labels_)
		delete data_labels_;

	data_labels_ = new CVisionData(labels.data(), labels.getType());
}

void CVisionModule::clearDataBuffers()
{
	for(vector<CVisionData*>::iterator buf_it = data_buffers_.begin(); buf_it != data_buffers_.end(); ++buf_it)
	{
		if(*buf_it)
			delete *buf_it;

		*buf_it = NULL;
	}
}

void CVisionModule::train()
{
	cout << "nothing to train here in base class... have you made an implementation mistake?" << endl;
}

void CVisionModule::setModuleID(const int& module_id)
{
	stringstream ss_mod_id;
	ss_mod_id << "module-" << module_id;
	module_id_ = ss_mod_id.str();
}
