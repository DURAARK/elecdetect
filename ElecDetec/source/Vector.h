#pragma once
#include "VisionData.h"

#include <vector>

template<class T>
class CVector :
	public CVisionData
{
public:
	inline CVector() { this->type_ = TYPE_VECTOR; }

	inline ~CVector() { }

	virtual void show()
	{
		std::cout << "This vector has " << vec_.size() << " elements." << std::endl;
	}

	std::vector<T> vec_;
};

