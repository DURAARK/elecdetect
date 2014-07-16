#pragma once

#include <iostream>


#define TYPE_SCALAR        1 << 0
#define TYPE_VECTOR        1 << 1
#define TYPE_MAT           1 << 2
#define TYPE_VECARRAY      1 << 3
#define TYPE_POINTVEC      1 << 4


class CVisionData
{
protected:
	int type_;

public:
	CVisionData() : type_(0) { };
	virtual ~CVisionData() { };

	virtual void show()
	{
		std::cout << "This is base class. Nothing to show." << std::endl;
	}

	inline int getType()
	{
		return type_;
	}
};

