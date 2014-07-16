#pragma once
#include "VisionData.h"

#include <vector>

template<class T>
class CVectorArray :
	public CVisionData
{
public:
	inline CVectorArray() { this->type_ = TYPE_VECARRAY; }
	inline ~CVectorArray() { }

	inline virtual void show()
	{
		std::cout << "This Vector-Arrary has " << array_.size() << " Elements";
		if (!array_.empty())
			std::cout << " (each of " << array_.front().size() << " length)";

		std::cout << "." << std::endl;
	}

	std::vector < std::vector<T> > array_;
};

