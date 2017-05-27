/* Copyright 2017 NVIDIA Corporation
 *
 * The U.S. Department of Energy funded the development of this software 
 * under subcontract B609478 with Lawrence Livermore National Security, LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <vector>
#include <cassert>

#ifndef __SNAP_CUDA_HELP_H__
#define __SNAP_CUDA_HELP_H__

__host__
static int gcd(int a, int b)
{
  if (a < b) return gcd(a, b-a);
  if (a > b) return gcd(a-b, b);
  return a;
}

template<int GROUPS, typename A, int N>
struct AccessorArray {
public:
  __host__
  AccessorArray(const std::vector<A> &accs)
  {
    assert(accs.size() == GROUPS);
    for (unsigned idx = 0; idx < GROUPS; idx++)
    {
      bases[idx] = accs[idx].base;
      if (idx == 0)
        strides = accs[0].strides;
      else
        assert(strides == accs[idx].strides);
    }
  }
public:
  __host__ __device__
  inline A operator[](unsigned idx) 
  { 
    A a;
    a.base = bases[idx];
    a.strides = strides;
    return a; 
  }
  __host__ __device__
  inline const A operator[](unsigned idx) const 
  { 
    A a;
    a.base = bases[idx];
    a.strides = strides;
    return a; 
  }
protected:
  intptr_t bases[GROUPS];
  // Deduplicate offsets here
  Point<N> strides;
};

template<int DIM, typename T>
struct ConstBuffer {
public:
  __host__
  ConstBuffer(const T buf[DIM])
  {
    for (int idx = 0; idx < DIM; idx++)
      buffer[idx] = buf[idx];
  }
  __host__
  ConstBuffer(const std::vector<T> &buf)
  {
    for (int idx = 0; idx < DIM; idx++)
      buffer[idx] = buf[idx];
  }
public:
  __host__ __device__ 
  inline T& operator[](unsigned idx) { return buffer[idx]; }
  __host__ __device__
  inline const T& operator[](unsigned idx) const { return buffer[idx]; }
public:
  T buffer[DIM];
};

#endif

