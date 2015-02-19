/*
 * RandomForest Implementation
 * Utils.h
 *
 *  Created on: Okt, 2014
 *      Author: Robert Viehauser
 *              robert(dot)viehauser(at)gmail(dot)com
 */


#ifndef UTILS_H
#define UTILS_H

#include <stdlib.h>
#include <math.h>

namespace RF
{
typedef unsigned int uint;

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

template <typename T>
inline T randRange(T low = 0, T high = 1)
{
    return (T)((((double)(high-low))*((double)rand()/RAND_MAX)))+low;
}

template <typename T>
inline void minLoc(const T* input_array, const uint& input_array_length, T& min, uint& min_loc, uint* considered_idxs = NULL, const uint& considered_idx_length = 0)
{
    if(!input_array || input_array_length == 0)
        return;

    uint c_idx_cnt = 0;

    // if considered_idx ptr is NULL, use all elements in input_array
    uint cur_idx = considered_idxs ? considered_idxs[c_idx_cnt] : 0;
    uint c_idx_length = considered_idxs ? considered_idx_length : input_array_length;

    for(min = input_array[cur_idx], min_loc = cur_idx; c_idx_cnt < c_idx_length; c_idx_cnt++)
    {
        // if considered_idx ptr is NULL, use all elements in input_array
        cur_idx = considered_idxs ? considered_idxs[c_idx_cnt] : c_idx_cnt;

        if(input_array[cur_idx] < min)
        {
            min = input_array[cur_idx];
            min_loc = cur_idx;
        }
    }
}

template <typename T>
inline void maxLoc(const T* input_array, const uint& input_array_length, T& max, uint& max_loc, uint* considered_idxs = NULL, const uint& considered_idx_length = 0)
{
    if(!input_array || input_array_length == 0)
        return;

    uint c_idx_cnt = 0;

    // if considered_idx ptr is NULL, use all elements in input_array
    uint cur_idx = considered_idxs ? considered_idxs[c_idx_cnt] : 0;
    uint c_idx_length = considered_idxs ? considered_idx_length : input_array_length;

    for(max = input_array[cur_idx], max_loc = cur_idx; c_idx_cnt < c_idx_length; c_idx_cnt++)
    {
        // if considered_idx ptr is NULL, use all elements in input_array
        cur_idx = considered_idxs ? considered_idxs[c_idx_cnt] : c_idx_cnt;

        if(input_array[cur_idx] > max)
        {
            max = input_array[cur_idx];
            max_loc = cur_idx;
        }
    }
}

template <typename T>
inline void minMaxLoc(const T* input_array, const uint& input_array_length, T& min, uint& min_loc, T& max, uint& max_loc, uint* considered_idxs = NULL, const uint& considered_idx_length = 0)
{
    if(!input_array || input_array_length == 0)
        return;

    uint c_idx_cnt = 0;

    // if considered_idx ptr is NULL, use all elements in input_array
    uint cur_idx = considered_idxs ? considered_idxs[c_idx_cnt] : 0;
    uint c_idx_length = considered_idxs ? considered_idx_length : input_array_length;

    for(min = input_array[cur_idx], max = input_array[cur_idx], min_loc = cur_idx, max_loc = cur_idx; c_idx_cnt < c_idx_length; c_idx_cnt++)
    {
        // if considered_idx ptr is NULL, use all elements in input_array
        cur_idx = considered_idxs ? considered_idxs[c_idx_cnt] : c_idx_cnt;

        if(input_array[cur_idx] > max)
        {
            max = input_array[cur_idx];
            max_loc = cur_idx;
        }
        else if(input_array[cur_idx] < min)
        {
            min = input_array[cur_idx];
            min_loc = cur_idx;
        }
    }
}

// change from sparse set definition vector to dense set representation
// i.e.: sparse: 2, 0, 0, 1, 2, 4  -> dense: 0, 0, 3, 4, 4, 5, 5, 5, 5
inline void sampleOccurangeToDirectSubsetIndices(const uint* const sparse_rep_vec, const uint& sparse_rep_length, uint* &dense_rep_vec, uint& dense_rep_length)
{
    // cumsum of sparse vector
    uint d_length = 0;
    for(uint sparse_idx = 0; sparse_idx < sparse_rep_length; ++sparse_idx)
    {
        d_length += sparse_rep_vec[sparse_idx];
    }
    uint* d_vec = Malloc(uint, d_length);
    for(uint sparse_idx = 0, d_idx = 0; sparse_idx < sparse_rep_length; ++sparse_idx)
    {
        if(sparse_rep_vec[sparse_idx] != 0) // speedup?
        {
            for(uint i = 0; i < sparse_rep_vec[sparse_idx]; ++i)
            {
                d_vec[d_idx] = sparse_idx;
                ++d_idx;
            }
        }
    }

    dense_rep_vec = d_vec;
    dense_rep_length = d_length;
}

// change from dense set representation to sparse set representation
// i.e.: dense: 0, 0, 3, 4, 4, 5, 5, 5, 5 -> sparse: 2, 0, 0, 1, 2, 4
//inline void denseIdxesToSparseIdxes(const uint* const dense_rep_vec, const uint& dense_rep_length, uint* &sparse_rep_vec, uint& sparse_rep_length)
//{
//    // preallocate space to hold the largest index:
//    // largest idx:
//    uint max_idx, max_loc;
//    uint min_idx, min_loc;
//    minMaxLoc<uint>(dense_rep_vec, dense_rep_length, min_idx, min_loc, max_idx, max_loc);

//    uint* s_vec = Malloc(uint, max_idx);
//    memset(s_vec, 0, sizeof(uint)*max_idx);

//    // make entries by counting
//    for(uint i = 0; i < dense_rep_vec; ++i)
//    {
//        ++s_vec[dense_rep_vec[i]];
//    }
//    sparse_rep_vec = s_vec;
//    sparse_rep_length = max_idx;
//}

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


} // namespace

#endif // UTILS_H
