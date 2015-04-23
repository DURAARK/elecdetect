/*
 * ElecDetec: Utils.h
 *
 *  Created on: Feb, 2015
 *      Author: Robert Viehauser
 */


#ifndef UTILS_H_
#define UTILS_H_


#include <string>
#include <sstream>
#include <vector>
#include <exception>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;

typedef unsigned int uint;

//--------------------------------------------
// split a string into parts according to a delimiter
inline vector<string> splitStringByDelimiter(const string& src0, const string& delimiter)
{
    vector<string> output;

    string src = src0; // make a non-const copy
    size_t pos = 0;
    string token;
    while ((pos = src.find(delimiter)) != string::npos)
    {
        token = src.substr(0, pos);
        output.push_back(token);
        src.erase(0, pos + delimiter.length());
    }
    output.push_back(src);

    return output;
}
//--------------------------------------------

#ifdef _MSC_VER
#define __func__ __FUNCTION__
#endif

// - Internal Assert -----------------------------------------
#define ELECDETEC_ASSERT(condition, msg) \
    if(!(condition)) \
    { \
    cerr << "ElecDetec Assert failed: " << #condition \
         << " exiting with code -1 (in file:" << __FILE__ << " line:" << __LINE__ << " function:" << __func__ << ")" << endl; \
    cerr << "Message: " << msg << endl; \
    exit(-1); \
    }
//------------------------------------------------------------




#define Malloc(type,n) (type *)malloc((n)*sizeof(type))



template <typename T>
inline T randRange(T low = 0, T high = 1)
{
    return (T)((((double)(high-low))*((double)rand()/RAND_MAX)))+low;
}


#define LINSPACE_DENSE -1
template <typename T>
inline void linspace(std::vector<T>& result, T start, T end, int N)
{
    result.clear();

    if(N == LINSPACE_DENSE)
    {
        for(T val = start; val <= end; val++)
            result.push_back(val);
        return;
    }

    if(N <= 1)
    {
        result.push_back(start);
        return;
    }

    for(int i = 0; i <= N-2; i++)
    {
        T cur_val = start + i*(end-start)/((T)N - 1.0);
        result.push_back(cur_val);
    }
    result.push_back(end);
}




inline Scalar getColorByIndex(const int& index)
{
    const Scalar_<uchar> color_model(80,240,0);
    return Scalar(color_model[2*index%3], color_model[(2*index+1)%3], color_model[(2*index+2)%3]);
}




template <typename T>
void quicksort_iterative(T* array, uint len)
{
   static const uint MAX = 64; /* stack size for max 2^(64/2) array elements  */
   uint left = 0, stack[MAX], pos = 0;
   for ( ; ; ) {                                           /* outer loop */
      for (; left+1 < len; len++) {                /* sort left to len-1 */
         if (pos == MAX) len = stack[pos = 0];  /* stack overflow, reset */
         uint rand_idx = randRange<uint>(left, len-1); /* pick random pivot */
         T pivot = array[rand_idx];
         stack[pos++] = len;                    /* sort right part later */
         for (unsigned right = left-1; ; ) { /* inner loop: partitioning */
            while (array[++right] < pivot);  /* look for greater element */
            while (pivot < array[--len]);    /* look for smaller element */
            if (right >= len) break;           /* partition point found? */
            T temp = array[right];
            array[right] = array[len];                  /* the only swap */
            array[len] = temp;
         }                            /* partitioned, continue left part */
      }
      if (pos == 0) break;                               /* stack empty? */
      left = len;                             /* left to right is sorted */
      len = stack[--pos];                      /* get next range to sort */
   }
}

// Unique: extract from a given array all different elements
template <typename T>
inline void uniqueSortedElements(const T* const input_vec, const uint& input_vec_length, T* &unique_vec, uint& unique_vec_length)
{
    // sort the input vector
    T* sorted_vec = Malloc(T, input_vec_length);
    memcpy(sorted_vec, input_vec, sizeof(T)*input_vec_length);
    quicksort_iterative<T>(sorted_vec, input_vec_length);

    // get enough space for the worst case, i.e. all elements are different
    T* u_vec = Malloc(T, input_vec_length);
    uint u_idx = 0;
    // get the unique elements
    T last_element = sorted_vec[0];
    u_vec[0] = last_element;
    for(uint i = 1; i < input_vec_length; i++)
    {
        if(last_element != sorted_vec[i])
        {
            u_vec[++u_idx] = sorted_vec[i];
            last_element = sorted_vec[i];
        }
    }
    free(sorted_vec);
    sorted_vec = NULL;

    // resize memory and return
    unique_vec = (T*)realloc(u_vec, sizeof(T)*(u_idx+1));
    unique_vec_length = u_idx+1;
}


#endif /* UTILS_H_ */
