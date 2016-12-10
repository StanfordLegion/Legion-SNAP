/* Copyright 2016 NVIDIA Corporation
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

#include "snap_types.h"
#include "accessor.h"

#include <vector>

using namespace LegionRuntime::Accessor;

template<int GROUPS>
__global__
void gpu_expand_cross_section(const double *sig_ptrs[GROUPS],
                              const int    *mat_ptr,
                                    double *xs_ptrs[GROUPS],
                              const ByteOffset sig_offsets[1],
                              const ByteOffset mat_offsets[3],
                              const ByteOffset xs_offsets[3],
                              const Point<3> origin)
{
  const int x = origin.x[0] + (blockIdx.x * blockDim.x + threadIdx.x);
  const int y = origin.x[1] + (blockIdx.y * blockDim.y + threadIdx.y);
  const int z = origin.x[2] + (blockIdx.z * blockDim.z + threadIdx.z);

  const int mat = *(mat_ptr + x * mat_offsets[0] + 
                              y * mat_offsets[1] + z * mat_offsets[2]);
  #pragma unroll
  for (int g = 0; g < GROUPS; g++) {
    const double *sig_ptr = sig_ptrs[g] + mat * sig_offsets[0];
    double val;
    asm volatile("ld.global.cs.f64 %0, [%1];" : "=d"(val) : "l"(sig_ptr) : "memory");
    double *xs_ptr = xs_ptrs[g] + x * xs_offsets[0] +
                                  y * xs_offsets[1] + z * xs_offsets[2];
    asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(xs_ptr), "d"(val) : "memory");
  }
}

template<int GROUPS>
__host__
void launch_expand_cross_section(const std::vector<double*> &sig_ptrs,
                                 const int *mat_ptr,
                                 const std::vector<double*> &xs_ptrs,
                                 const ByteOffset sig_offsets[1],
                                 const ByteOffset mat_offsets[3],
                                 const ByteOffset xs_offsets[3],
                                 const Point<3> &origin,
                                 const dim3 &grid, const dim3 &block)
{
  const double *sig_buffer[GROUPS];
  for (int g = 0; g < GROUPS; g++)
    sig_buffer[g] = sig_ptrs[g];
  double *xs_buffer[GROUPS];
  for (int g = 0; g < GROUPS; g++)
    xs_buffer[g] = xs_ptrs[g];
  
  gpu_expand_cross_section<GROUPS><<<grid, block>>>(sig_buffer, mat_ptr, xs_buffer,
                                                    sig_offsets, mat_offsets,
                                                    xs_offsets, origin);
}

__host__
void run_expand_cross_section(const std::vector<double*> &sig_ptrs,
                              const int *mat_ptr,
                              const std::vector<double*> &xs_ptrs,
                              const ByteOffset sig_offsets[1],
                              const ByteOffset mat_offsets[3],
                              const ByteOffset xs_offsets[3],
                              const Rect<3> &subgrid_bounds)
{
  // Figure out the dimensions to launch
  const int x_range = (subgrid_bounds.hi[0] - subgrid_bounds.lo[0]) + 1; 
  const int y_range = (subgrid_bounds.hi[1] - subgrid_bounds.lo[1]) + 1;
  const int z_range = (subgrid_bounds.hi[2] - subgrid_bounds.lo[2]) + 1;

  assert((x_range % 8) == 0);
  assert((y_range % 4) == 0);
  assert((z_range % 4) == 0);

  dim3 grid(x_range/8, y_range/4, z_range/4);
  dim3 block(8, 4, 4);

  // Switch on the number of groups
  assert(sig_ptrs.size() == xs_ptrs.size());
  // TODO: replace this template foolishness with Terra
  switch (sig_ptrs.size())
  {
    case 1:
      {
        launch_expand_cross_section<1>(sig_ptrs, mat_ptr, xs_ptrs,
                                       sig_offsets, mat_offsets, 
                                       xs_offsets, subgrid_bounds.lo,
                                       grid, block);
        break;
      }
    case 2:
      {
        launch_expand_cross_section<2>(sig_ptrs, mat_ptr, xs_ptrs,
                                       sig_offsets, mat_offsets, 
                                       xs_offsets, subgrid_bounds.lo,
                                       grid, block);
        break;
      }
    case 3:
      {
        launch_expand_cross_section<3>(sig_ptrs, mat_ptr, xs_ptrs,
                                       sig_offsets, mat_offsets, 
                                       xs_offsets, subgrid_bounds.lo,
                                       grid, block);
        break;
      }
    case 4:
      {
        launch_expand_cross_section<4>(sig_ptrs, mat_ptr, xs_ptrs,
                                       sig_offsets, mat_offsets, 
                                       xs_offsets, subgrid_bounds.lo,
                                       grid, block);
        break;
      }
    case 5:
      {
        launch_expand_cross_section<5>(sig_ptrs, mat_ptr, xs_ptrs,
                                       sig_offsets, mat_offsets, 
                                       xs_offsets, subgrid_bounds.lo,
                                       grid, block);
        break;
      }
    case 6:
      {
        launch_expand_cross_section<6>(sig_ptrs, mat_ptr, xs_ptrs,
                                       sig_offsets, mat_offsets, 
                                       xs_offsets, subgrid_bounds.lo,
                                       grid, block);
        break;
      }
    case 7:
      {
        launch_expand_cross_section<7>(sig_ptrs, mat_ptr, xs_ptrs,
                                       sig_offsets, mat_offsets, 
                                       xs_offsets, subgrid_bounds.lo,
                                       grid, block);
        break;
      }
    case 8:
      {
        launch_expand_cross_section<8>(sig_ptrs, mat_ptr, xs_ptrs,
                                       sig_offsets, mat_offsets, 
                                       xs_offsets, subgrid_bounds.lo,
                                       grid, block);
        break;
      }
    case 9:
      {
        launch_expand_cross_section<9>(sig_ptrs, mat_ptr, xs_ptrs,
                                       sig_offsets, mat_offsets, 
                                       xs_offsets, subgrid_bounds.lo,
                                       grid, block);
        break;
      }
    case 10:
      {
        launch_expand_cross_section<10>(sig_ptrs, mat_ptr, xs_ptrs,
                                       sig_offsets, mat_offsets, 
                                       xs_offsets, subgrid_bounds.lo,
                                       grid, block);
        break;
      }
    case 11:
      {
        launch_expand_cross_section<11>(sig_ptrs, mat_ptr, xs_ptrs,
                                       sig_offsets, mat_offsets, 
                                       xs_offsets, subgrid_bounds.lo,
                                       grid, block);
        break;
      }
    case 12:
      {
        launch_expand_cross_section<12>(sig_ptrs, mat_ptr, xs_ptrs,
                                       sig_offsets, mat_offsets, 
                                       xs_offsets, subgrid_bounds.lo,
                                       grid, block);
        break;
      }
    case 13:
      {
        launch_expand_cross_section<13>(sig_ptrs, mat_ptr, xs_ptrs,
                                       sig_offsets, mat_offsets, 
                                       xs_offsets, subgrid_bounds.lo,
                                       grid, block);
        break;
      }
    case 14:
      {
        launch_expand_cross_section<14>(sig_ptrs, mat_ptr, xs_ptrs,
                                       sig_offsets, mat_offsets, 
                                       xs_offsets, subgrid_bounds.lo,
                                       grid, block);
        break;
      }
    case 15:
      {
        launch_expand_cross_section<15>(sig_ptrs, mat_ptr, xs_ptrs,
                                       sig_offsets, mat_offsets, 
                                       xs_offsets, subgrid_bounds.lo,
                                       grid, block);
        break;
      }
    case 16:
      {
        launch_expand_cross_section<16>(sig_ptrs, mat_ptr, xs_ptrs,
                                       sig_offsets, mat_offsets, 
                                       xs_offsets, subgrid_bounds.lo,
                                       grid, block);
        break;
      }
    default:
      assert(false); // add more cases
  }
}

template<int GROUPS>
__global__
void gpu_expand_scattering_cross_section(const MomentQuad *slgg_ptrs[GROUPS],
                                         const int        *mat_ptr,
                                               MomentQuad *xs_ptrs[GROUPS],
                                         const ByteOffset slgg_offsets[2],
                                         const ByteOffset mat_offsets[3],
                                         const ByteOffset xs_offsets[3],
                                         const Point<3> origin,
                                         const int group_start)
{
  const int x = origin.x[0] + (blockIdx.x * blockDim.x + threadIdx.x);
  const int y = origin.x[1] + (blockIdx.y * blockDim.y + threadIdx.y);
  const int z = origin.x[2] + (blockIdx.z * blockDim.z + threadIdx.z);

  const int mat = *(mat_ptr + x * mat_offsets[0] + 
                              y * mat_offsets[1] + z * mat_offsets[2]);
  #pragma unroll
  for (int g = 0; g < GROUPS; g++) {
    MomentQuad quad = *(slgg_ptrs[g] + mat * slgg_offsets[0] +
                        (group_start + g) * slgg_offsets[1]);
    *(xs_ptrs[g] + x * xs_offsets[0] + y * xs_offsets[1] +
        z * xs_offsets[2]) = quad;
  }
}

template<int GROUPS>
__host__
void launch_expand_scattering_cross_section(
                                      const std::vector<MomentQuad*> &slgg_ptrs,
                                      const int *mat_ptr,
                                      const std::vector<MomentQuad*> &xs_ptrs,
                                      const ByteOffset slgg_offsets[3],
                                      const ByteOffset mat_offsets[3],
                                      const ByteOffset xs_offsets[3],
                                      const Point<3> &origin,
                                      const int group_start,
                                      const dim3 &grid, const dim3 &block)
{
  const MomentQuad *slgg_buffer[GROUPS];
  for (int g = 0; g < GROUPS; g++)
    slgg_buffer[g] = slgg_ptrs[g];
  MomentQuad *xs_buffer[GROUPS];
  for (int g = 0; g < GROUPS; g++)
    xs_buffer[g] = xs_ptrs[g];

  gpu_expand_scattering_cross_section<GROUPS><<<grid,block>>>(slgg_buffer,
                                                    mat_ptr, xs_buffer,
                                                    slgg_offsets, mat_offsets,
                                                    xs_offsets, origin, group_start);
}

__host__
void run_expand_scattering_cross_section(
                                      const std::vector<MomentQuad*> &slgg_ptrs,
                                      const int *mat_ptr,
                                      const std::vector<MomentQuad*> &xs_ptrs,
                                      const ByteOffset slgg_offsets[3],
                                      const ByteOffset mat_offsets[3],
                                      const ByteOffset xs_offsets[3],
                                      const Rect<3> &subgrid_bounds,
                                      const int group_start)
{
  // Figure out the dimensions to launch
  const int x_range = (subgrid_bounds.hi[0] - subgrid_bounds.lo[0]) + 1; 
  const int y_range = (subgrid_bounds.hi[1] - subgrid_bounds.lo[1]) + 1;
  const int z_range = (subgrid_bounds.hi[2] - subgrid_bounds.lo[2]) + 1;

  assert((x_range % 8) == 0);
  assert((y_range % 4) == 0);
  assert((z_range % 4) == 0);

  dim3 grid(x_range/8, y_range/4, z_range/4);
  dim3 block(8, 4, 4);

  // Switch on the number of groups
  assert(slgg_ptrs.size() == xs_ptrs.size());
  // TODO: replace this template foolishness with Terra
  switch (slgg_ptrs.size())
  {
    case 1:
      {
        launch_expand_scattering_cross_section<1>(slgg_ptrs, mat_ptr, xs_ptrs,
                                                  slgg_offsets, mat_offsets,
                                                  xs_offsets, subgrid_bounds.lo,
                                                  group_start, grid, block);
        break;
      }
    case 2:
      {
        launch_expand_scattering_cross_section<2>(slgg_ptrs, mat_ptr, xs_ptrs,
                                                  slgg_offsets, mat_offsets,
                                                  xs_offsets, subgrid_bounds.lo,
                                                  group_start, grid, block);
        break;
      }
    case 3:
      {
        launch_expand_scattering_cross_section<3>(slgg_ptrs, mat_ptr, xs_ptrs,
                                                  slgg_offsets, mat_offsets,
                                                  xs_offsets, subgrid_bounds.lo,
                                                  group_start, grid, block);
        break;
      }
    case 4:
      {
        launch_expand_scattering_cross_section<4>(slgg_ptrs, mat_ptr, xs_ptrs,
                                                  slgg_offsets, mat_offsets,
                                                  xs_offsets, subgrid_bounds.lo,
                                                  group_start, grid, block);
        break;
      }
    case 5:
      {
        launch_expand_scattering_cross_section<5>(slgg_ptrs, mat_ptr, xs_ptrs,
                                                  slgg_offsets, mat_offsets,
                                                  xs_offsets, subgrid_bounds.lo,
                                                  group_start, grid, block);
        break;
      }
    case 6:
      {
        launch_expand_scattering_cross_section<6>(slgg_ptrs, mat_ptr, xs_ptrs,
                                                  slgg_offsets, mat_offsets,
                                                  xs_offsets, subgrid_bounds.lo,
                                                  group_start, grid, block);
        break;
      }
    case 7:
      {
        launch_expand_scattering_cross_section<7>(slgg_ptrs, mat_ptr, xs_ptrs,
                                                  slgg_offsets, mat_offsets,
                                                  xs_offsets, subgrid_bounds.lo,
                                                  group_start, grid, block);
        break;
      }
    case 8:
      {
        launch_expand_scattering_cross_section<8>(slgg_ptrs, mat_ptr, xs_ptrs,
                                                  slgg_offsets, mat_offsets,
                                                  xs_offsets, subgrid_bounds.lo,
                                                  group_start, grid, block);
        break;
      }
    case 9:
      {
        launch_expand_scattering_cross_section<9>(slgg_ptrs, mat_ptr, xs_ptrs,
                                                  slgg_offsets, mat_offsets,
                                                  xs_offsets, subgrid_bounds.lo,
                                                  group_start, grid, block);
        break;
      }
    case 10:
      {
        launch_expand_scattering_cross_section<10>(slgg_ptrs, mat_ptr, xs_ptrs,
                                                  slgg_offsets, mat_offsets,
                                                  xs_offsets, subgrid_bounds.lo,
                                                  group_start, grid, block);
        break;
      }
    case 11:
      {
        launch_expand_scattering_cross_section<11>(slgg_ptrs, mat_ptr, xs_ptrs,
                                                  slgg_offsets, mat_offsets,
                                                  xs_offsets, subgrid_bounds.lo,
                                                  group_start, grid, block);
        break;
      }
    case 12:
      {
        launch_expand_scattering_cross_section<12>(slgg_ptrs, mat_ptr, xs_ptrs,
                                                  slgg_offsets, mat_offsets,
                                                  xs_offsets, subgrid_bounds.lo,
                                                  group_start, grid, block);
        break;
      }
    case 13:
      {
        launch_expand_scattering_cross_section<13>(slgg_ptrs, mat_ptr, xs_ptrs,
                                                  slgg_offsets, mat_offsets,
                                                  xs_offsets, subgrid_bounds.lo,
                                                  group_start, grid, block);
        break;
      }
    case 14:
      {
        launch_expand_scattering_cross_section<14>(slgg_ptrs, mat_ptr, xs_ptrs,
                                                  slgg_offsets, mat_offsets,
                                                  xs_offsets, subgrid_bounds.lo,
                                                  group_start, grid, block);
        break;
      }
    case 15:
      {
        launch_expand_scattering_cross_section<15>(slgg_ptrs, mat_ptr, xs_ptrs,
                                                  slgg_offsets, mat_offsets,
                                                  xs_offsets, subgrid_bounds.lo,
                                                  group_start, grid, block);
        break;
      }
    case 16:
      {
        launch_expand_scattering_cross_section<16>(slgg_ptrs, mat_ptr, xs_ptrs,
                                                  slgg_offsets, mat_offsets,
                                                  xs_offsets, subgrid_bounds.lo,
                                                  group_start, grid, block);
        break;
      }
    default:
      assert(false); // add more cases
  }
}

