//#pragma once
//#include "VisionData.h"
//
//#include <vector>
//
//
//
//template<class T>
//class CScalar :
//	public CVisionData
//{
//public:
//	inline CScalar() { this->type_ = TYPE_SCALAR; }
//	inline CScalar(T value)
//	{
//		this->type_ = TYPE_SCALAR;
//		this->val_ = value;
//	}
//	virtual ~CScalar() { }
//
//	virtual void show()
//	{
//		std::cout << "Value of this scalar is " << val_ << "." << std::endl;
//	}
//
//	T val_;
//};
//
//
//
//template<class T>
//class CWeightedScalar :
//	public CScalar<T>
//{
//public:
//	inline CWeightedScalar() : weight_(1.0)
//	{
//		this->val_ = 0;
//	}
//	inline CWeightedScalar(T value, float weight = 1.0f)
//	{
//		this->val_ = value;
//		this->weight_ = weight;
//	}
//	virtual ~CWeightedScalar() { }
//
//	virtual void show()
//	{
//		std::cout << "Value of this weighted scalar is " << this->val_ << " and its weight: " << weight_ << std::endl;
//	}
//
//	float weight_;
//};
//
//
//
