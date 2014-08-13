/*
 * Exceptions.h
 *
 *  Created on: Jun 24, 2014
 *      Author: test
 */

#ifndef EXCEPTIONS_H_
#define EXCEPTIONS_H_

#include <sstream>
#include <exception>

using namespace std;

// is thrown if command parameter are faulty
class ParamExecption : public exception
{
private:
	const char* reason_;
public:
	//ParamExecption(){ /*reason_ = "no reason";*/ };
	ParamExecption(const char* reason) : reason_(reason) { };

	virtual const char* what() const throw()
	{
		stringstream msg;
		msg << "Parameter exception happened! Reason: " << reason_ << endl;
		return msg.str().c_str();
	}
};

// is thrown if something went wrong on pipeline setup
class PipeConfigExecption : public exception
{
public:
	PipeConfigExecption() { };

	virtual const char* what() const throw()
	{
		stringstream msg;
		msg << "Pipe config exception happened! Reason: " << endl;
		return msg.str().c_str();
	}
};

// is thrown if VisionModule is executed before training
class NotTrainedExecption : public exception
{
private:
	const char* reason_;
public:
	//ParamExecption(){ /*reason_ = "no reason";*/ };
	NotTrainedExecption(const char* reason) : reason_(reason) { };

	virtual const char* what() const throw()
	{
		stringstream msg;
		msg << "NotTrained exception happened! Reason: " << reason_ << endl;
		return msg.str().c_str();
	}
};

// is thrown if Vision Module can't handle type of last Data object
class VisionDataTypeException : public exception
{
private:
	VisionDataTypeException() : is_type_(0), should_type_(0) { }
	int is_type_, should_type_;

public:
	VisionDataTypeException(int is_type, int should_type) : is_type_(is_type), should_type_(should_type)
	{
		//cout << "New TypeExeption created: is:" << is_type_ << " should:" << should_type_ << endl;
	}

	virtual const char* what() const throw()
	{
		stringstream msg;
		msg << "Type exception happened! Type is: " << is_type_ << " but should be: " << should_type_ << endl;
		return msg.str().c_str();
	}
};

// is thrown if the size of last Data object if faulty
class VisionDataSizeException : public exception
{
private:
	VisionDataSizeException() : is_size_(0), should_size_(0) { }
	int is_size_, should_size_;

public:
	VisionDataSizeException(int is_type, int should_type) : is_size_(is_type), should_size_(should_type)
	{
		//cout << "New TypeExeption created: is:" << is_size_ << " should:" << should_size_ << endl;
	}

	virtual const char* what() const throw()
	{
		stringstream msg;
		msg << "Size exception happened! Wrong size is: " << is_size_ << " but should be: " << should_size_ << endl;
		return msg.str().c_str();
	}
};



#endif /* EXCEPTIONS_H_ */
