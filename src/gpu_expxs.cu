/* Copyright 2017 NVIDIA Corporation
 *
 * The U.S. Department of Energy funded the development of this software 
 * under subcontract B609478 with Lawrence Livermore National Security, LLC
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

#include "snap.h"
#include "snap_cuda_help.h"

template<int GROUPS>
__global__
void gpu_expand_cross_section(const Point<3> origin,
                              const AccessorArray<GROUPS,
                                      Accessor<double,1>,1> fa_sig,
                              const Accessor<int,3> fa_mat,
                                    AccessorArray<GROUPS,
                                      Accessor<double,3>,3> fa_xs)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;
  const Point<3> p = origin + Point<3>(x,y,z);

  const int mat = fa_mat[p];
  #pragma unroll
  for (int g = 0; g < GROUPS; g++) {
    const double *sig_ptr = fa_sig[g].ptr(mat);
    double val;
    asm volatile("ld.global.cs.f64 %0, [%1];" : "=d"(val) : "l"(sig_ptr) : "memory");
    double *xs_ptr = fa_xs[g].ptr(p);
    asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(xs_ptr), "d"(val) : "memory");
  }
}

__host__
void run_expand_cross_section(const std::vector<Accessor<double,1> > &fa_sig,
                              const Accessor<int,3> &fa_mat,
                              const std::vector<Accessor<double,3> > &fa_xs,
                              const Rect<3> &subgrid_bounds)
{
  // Figure out the dimensions to launch
  const int x_range = (subgrid_bounds.hi[0] - subgrid_bounds.lo[0]) + 1; 
  const int y_range = (subgrid_bounds.hi[1] - subgrid_bounds.lo[1]) + 1;
  const int z_range = (subgrid_bounds.hi[2] - subgrid_bounds.lo[2]) + 1;

  dim3 block(gcd(x_range,32), gcd(y_range,4), gcd(z_range,4));
  dim3 grid(x_range/block.x, y_range/block.y, z_range/block.z);

  // Switch on the number of groups
  assert(fa_sig.size() == fa_xs.size());
  // TODO: replace this template foolishness with Terra
  switch (fa_sig.size())
  {
    case 1:
      {
        gpu_expand_cross_section<1><<<grid, block>>>(subgrid_bounds.lo,
                                       AccessorArray<1,
                                        Accessor<double,1>,1>(fa_sig), fa_mat,
                                       AccessorArray<1,
                                        Accessor<double,3>,3>(fa_xs));
        break;
      }
    case 2:
      {
        gpu_expand_cross_section<2><<<grid, block>>>(subgrid_bounds.lo,
                                       AccessorArray<2,
                                        Accessor<double,1>,1>(fa_sig), fa_mat,
                                       AccessorArray<2,
                                        Accessor<double,3>,3>(fa_xs));
        break;
      }
    case 3:
      {
        gpu_expand_cross_section<3><<<grid, block>>>(subgrid_bounds.lo,
                                       AccessorArray<3,
                                        Accessor<double,1>,1>(fa_sig), fa_mat,
                                       AccessorArray<3,
                                        Accessor<double,3>,3>(fa_xs));
        break;
      }
    case 4:
      {
        gpu_expand_cross_section<4><<<grid, block>>>(subgrid_bounds.lo,
                                       AccessorArray<4,
                                        Accessor<double,1>,1>(fa_sig), fa_mat,
                                       AccessorArray<4,
                                        Accessor<double,3>,3>(fa_xs));
        break;
      }
    case 5:
      {
        gpu_expand_cross_section<5><<<grid, block>>>(subgrid_bounds.lo,
                                       AccessorArray<5,
                                        Accessor<double,1>,1>(fa_sig), fa_mat,
                                       AccessorArray<5,
                                        Accessor<double,3>,3>(fa_xs));
        break;
      }
    case 6:
      {
        gpu_expand_cross_section<6><<<grid, block>>>(subgrid_bounds.lo,
                                       AccessorArray<6,
                                        Accessor<double,1>,1>(fa_sig), fa_mat,
                                       AccessorArray<6,
                                        Accessor<double,3>,3>(fa_xs));
        break;
      }
    case 7:
      {
        gpu_expand_cross_section<7><<<grid, block>>>(subgrid_bounds.lo,
                                       AccessorArray<7,
                                        Accessor<double,1>,1>(fa_sig), fa_mat,
                                       AccessorArray<7,
                                        Accessor<double,3>,3>(fa_xs));
        break;
      }
    case 8:
      {
        gpu_expand_cross_section<8><<<grid, block>>>(subgrid_bounds.lo,
                                       AccessorArray<8,
                                        Accessor<double,1>,1>(fa_sig), fa_mat,
                                       AccessorArray<8,
                                        Accessor<double,3>,3>(fa_xs));
        break;
      }
    case 9:
      {
        gpu_expand_cross_section<9><<<grid, block>>>(subgrid_bounds.lo,
                                       AccessorArray<9,
                                        Accessor<double,1>,1>(fa_sig), fa_mat,
                                       AccessorArray<9,
                                        Accessor<double,3>,3>(fa_xs));
        break;
      }
    case 10:
      {
        gpu_expand_cross_section<10><<<grid, block>>>(subgrid_bounds.lo,
                                       AccessorArray<10,
                                        Accessor<double,1>,1>(fa_sig), fa_mat,
                                       AccessorArray<10,
                                        Accessor<double,3>,3>(fa_xs));
        break;
      }
    case 11:
      {
        gpu_expand_cross_section<11><<<grid, block>>>(subgrid_bounds.lo,
                                       AccessorArray<11,
                                        Accessor<double,1>,1>(fa_sig), fa_mat,
                                       AccessorArray<11,
                                        Accessor<double,3>,3>(fa_xs));
        break;
      }
    case 12:
      {
        gpu_expand_cross_section<12><<<grid, block>>>(subgrid_bounds.lo,
                                       AccessorArray<12,
                                        Accessor<double,1>,1>(fa_sig), fa_mat,
                                       AccessorArray<12,
                                        Accessor<double,3>,3>(fa_xs));
        break;
      }
    case 13:
      {
        gpu_expand_cross_section<13><<<grid, block>>>(subgrid_bounds.lo,
                                       AccessorArray<13,
                                        Accessor<double,1>,1>(fa_sig), fa_mat,
                                       AccessorArray<13,
                                        Accessor<double,3>,3>(fa_xs));
        break;
      }
    case 14:
      {
        gpu_expand_cross_section<14><<<grid, block>>>(subgrid_bounds.lo,
                                       AccessorArray<14,
                                        Accessor<double,1>,1>(fa_sig), fa_mat,
                                       AccessorArray<14,
                                        Accessor<double,3>,3>(fa_xs));
        break;
      }
    case 15:
      {
        gpu_expand_cross_section<15><<<grid, block>>>(subgrid_bounds.lo,
                                       AccessorArray<15,
                                        Accessor<double,1>,1>(fa_sig), fa_mat,
                                       AccessorArray<15,
                                        Accessor<double,3>,3>(fa_xs));
        break;
      }
    case 16:
      {
        gpu_expand_cross_section<16><<<grid, block>>>(subgrid_bounds.lo,
                                       AccessorArray<16,
                                        Accessor<double,1>,1>(fa_sig), fa_mat,
                                       AccessorArray<16,
                                        Accessor<double,3>,3>(fa_xs));
        break;
      }
    case 24:
      {
        gpu_expand_cross_section<24><<<grid, block>>>(subgrid_bounds.lo,
                                       AccessorArray<24,
                                        Accessor<double,1>,1>(fa_sig), fa_mat,
                                       AccessorArray<24,
                                        Accessor<double,3>,3>(fa_xs));
        break;
      }
    case 32:
      {
        gpu_expand_cross_section<32><<<grid, block>>>(subgrid_bounds.lo,
                                       AccessorArray<32,
                                        Accessor<double,1>,1>(fa_sig), fa_mat,
                                       AccessorArray<32,
                                        Accessor<double,3>,3>(fa_xs));
        break;
      }
    case 40:
      {
        gpu_expand_cross_section<40><<<grid, block>>>(subgrid_bounds.lo,
                                       AccessorArray<40,
                                        Accessor<double,1>,1>(fa_sig), fa_mat,
                                       AccessorArray<40,
                                        Accessor<double,3>,3>(fa_xs));
        break;
      }
    case 48:
      {
        gpu_expand_cross_section<48><<<grid, block>>>(subgrid_bounds.lo,
                                       AccessorArray<48,
                                        Accessor<double,1>,1>(fa_sig), fa_mat,
                                       AccessorArray<48,
                                        Accessor<double,3>,3>(fa_xs));
        break;
      }
    case 56:
      {
        gpu_expand_cross_section<56><<<grid, block>>>(subgrid_bounds.lo,
                                       AccessorArray<56,
                                        Accessor<double,1>,1>(fa_sig), fa_mat,
                                       AccessorArray<56,
                                        Accessor<double,3>,3>(fa_xs));
        break;
      }
    case 64:
      {
        gpu_expand_cross_section<64><<<grid, block>>>(subgrid_bounds.lo,
                                       AccessorArray<64,
                                        Accessor<double,1>,1>(fa_sig), fa_mat,
                                       AccessorArray<64,
                                        Accessor<double,3>,3>(fa_xs));
        break;
      }
    default:
      assert(false); // add more cases
  }
}

template<int GROUPS>
__global__
void gpu_expand_scattering_cross_section(const Point<3> origin,
                                         const AccessorArray<GROUPS,
                                                Accessor<MomentQuad,2>,2> fa_slgg,
                                         const Accessor<int,3> fa_mat,
                                               AccessorArray<GROUPS,
                                                Accessor<MomentQuad,3>,3> fa_xs,
                                         const int group_start)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;
  const Point<3> p = origin + Point<3>(x,y,z);

  const int mat = fa_mat[p];
  #pragma unroll
  for (int g = 0; g < GROUPS; g++)
    fa_xs[g][p] = fa_slgg[g][Point<2>(mat,group_start+g)];
}

__host__
void run_expand_scattering_cross_section(
                                      const std::vector<Accessor<MomentQuad,2> > &fa_slgg,
                                      const Accessor<int,3> &fa_mat,
                                      const std::vector<Accessor<MomentQuad,3> > &fa_xs,
                                      const Rect<3> &subgrid_bounds,
                                      const int group_start)
{
  // Figure out the dimensions to launch
  const int x_range = (subgrid_bounds.hi[0] - subgrid_bounds.lo[0]) + 1; 
  const int y_range = (subgrid_bounds.hi[1] - subgrid_bounds.lo[1]) + 1;
  const int z_range = (subgrid_bounds.hi[2] - subgrid_bounds.lo[2]) + 1;

  dim3 block(gcd(x_range,32),gcd(y_range,4),gcd(z_range,4));
  dim3 grid(x_range/block.x, y_range/block.y, z_range/block.z);

  // Switch on the number of groups
  assert(fa_slgg.size() == fa_xs.size());
  // TODO: replace this template foolishness with Terra
  switch (fa_slgg.size())
  {
    case 1:
      {
        gpu_expand_scattering_cross_section<1><<<grid,block>>>(
                            subgrid_bounds.lo,
                            AccessorArray<1,Accessor<MomentQuad,2>,2>(fa_slgg),
                            fa_mat,
                            AccessorArray<1,Accessor<MomentQuad,3>,3>(fa_xs),
                            group_start);
        break;
      }
    case 2:
      {
        gpu_expand_scattering_cross_section<2><<<grid,block>>>(
                            subgrid_bounds.lo,
                            AccessorArray<2,Accessor<MomentQuad,2>,2>(fa_slgg),
                            fa_mat,
                            AccessorArray<2,Accessor<MomentQuad,3>,3>(fa_xs),
                            group_start);
        break;
      }
    case 3:
      {
        gpu_expand_scattering_cross_section<3><<<grid,block>>>(
                            subgrid_bounds.lo,
                            AccessorArray<3,Accessor<MomentQuad,2>,2>(fa_slgg),
                            fa_mat,
                            AccessorArray<3,Accessor<MomentQuad,3>,3>(fa_xs),
                            group_start);
        break;
      }
    case 4:
      {
        gpu_expand_scattering_cross_section<4><<<grid,block>>>(
                            subgrid_bounds.lo,
                            AccessorArray<4,Accessor<MomentQuad,2>,2>(fa_slgg),
                            fa_mat,
                            AccessorArray<4,Accessor<MomentQuad,3>,3>(fa_xs),
                            group_start);
        break;
      }
    case 5:
      {
        gpu_expand_scattering_cross_section<5><<<grid,block>>>(
                            subgrid_bounds.lo,
                            AccessorArray<5,Accessor<MomentQuad,2>,2>(fa_slgg),
                            fa_mat,
                            AccessorArray<5,Accessor<MomentQuad,3>,3>(fa_xs),
                            group_start);
        break;
      }
    case 6:
      {
        gpu_expand_scattering_cross_section<6><<<grid,block>>>(
                            subgrid_bounds.lo,
                            AccessorArray<6,Accessor<MomentQuad,2>,2>(fa_slgg),
                            fa_mat,
                            AccessorArray<6,Accessor<MomentQuad,3>,3>(fa_xs),
                            group_start);
        break;
      }
    case 7:
      {
        gpu_expand_scattering_cross_section<7><<<grid,block>>>(
                            subgrid_bounds.lo,
                            AccessorArray<7,Accessor<MomentQuad,2>,2>(fa_slgg),
                            fa_mat,
                            AccessorArray<7,Accessor<MomentQuad,3>,3>(fa_xs),
                            group_start);
        break;
      }
    case 8:
      {
        gpu_expand_scattering_cross_section<8><<<grid,block>>>(
                            subgrid_bounds.lo,
                            AccessorArray<8,Accessor<MomentQuad,2>,2>(fa_slgg),
                            fa_mat,
                            AccessorArray<8,Accessor<MomentQuad,3>,3>(fa_xs),
                            group_start);
        break;
      }
    case 9:
      {
        gpu_expand_scattering_cross_section<9><<<grid,block>>>(
                            subgrid_bounds.lo,
                            AccessorArray<9,Accessor<MomentQuad,2>,2>(fa_slgg),
                            fa_mat,
                            AccessorArray<9,Accessor<MomentQuad,3>,3>(fa_xs),
                            group_start);
        break;
      }
    case 10:
      {
        gpu_expand_scattering_cross_section<10><<<grid,block>>>(
                            subgrid_bounds.lo,
                            AccessorArray<10,Accessor<MomentQuad,2>,2>(fa_slgg),
                            fa_mat,
                            AccessorArray<10,Accessor<MomentQuad,3>,3>(fa_xs),
                            group_start);
        break;
      }
    case 11:
      {
        gpu_expand_scattering_cross_section<11><<<grid,block>>>(
                            subgrid_bounds.lo,
                            AccessorArray<11,Accessor<MomentQuad,2>,2>(fa_slgg),
                            fa_mat,
                            AccessorArray<11,Accessor<MomentQuad,3>,3>(fa_xs),
                            group_start);
        break;
      }
    case 12:
      {
        gpu_expand_scattering_cross_section<12><<<grid,block>>>(
                            subgrid_bounds.lo,
                            AccessorArray<12,Accessor<MomentQuad,2>,2>(fa_slgg),
                            fa_mat,
                            AccessorArray<12,Accessor<MomentQuad,3>,3>(fa_xs),
                            group_start);
        break;
      }
    case 13:
      {
        gpu_expand_scattering_cross_section<13><<<grid,block>>>(
                            subgrid_bounds.lo,
                            AccessorArray<13,Accessor<MomentQuad,2>,2>(fa_slgg),
                            fa_mat,
                            AccessorArray<13,Accessor<MomentQuad,3>,3>(fa_xs),
                            group_start);
        break;
      }
    case 14:
      {
        gpu_expand_scattering_cross_section<14><<<grid,block>>>(
                            subgrid_bounds.lo,
                            AccessorArray<14,Accessor<MomentQuad,2>,2>(fa_slgg),
                            fa_mat,
                            AccessorArray<14,Accessor<MomentQuad,3>,3>(fa_xs),
                            group_start);
        break;
      }
    case 15:
      {
        gpu_expand_scattering_cross_section<15><<<grid,block>>>(
                            subgrid_bounds.lo,
                            AccessorArray<15,Accessor<MomentQuad,2>,2>(fa_slgg),
                            fa_mat,
                            AccessorArray<15,Accessor<MomentQuad,3>,3>(fa_xs),
                            group_start);
        break;
      }
    case 16:
      {
        gpu_expand_scattering_cross_section<16><<<grid,block>>>(
                            subgrid_bounds.lo,
                            AccessorArray<16,Accessor<MomentQuad,2>,2>(fa_slgg),
                            fa_mat,
                            AccessorArray<16,Accessor<MomentQuad,3>,3>(fa_xs),
                            group_start);
        break;
      }
    case 24:
      {
        gpu_expand_scattering_cross_section<24><<<grid,block>>>(
                            subgrid_bounds.lo,
                            AccessorArray<24,Accessor<MomentQuad,2>,2>(fa_slgg),
                            fa_mat,
                            AccessorArray<24,Accessor<MomentQuad,3>,3>(fa_xs),
                            group_start);
        break;
      }
    case 32:
      {
        gpu_expand_scattering_cross_section<32><<<grid,block>>>(
                            subgrid_bounds.lo,
                            AccessorArray<32,Accessor<MomentQuad,2>,2>(fa_slgg),
                            fa_mat,
                            AccessorArray<32,Accessor<MomentQuad,3>,3>(fa_xs),
                            group_start);
        break;
      }
    case 40:
      {
        gpu_expand_scattering_cross_section<40><<<grid,block>>>(
                            subgrid_bounds.lo,
                            AccessorArray<40,Accessor<MomentQuad,2>,2>(fa_slgg),
                            fa_mat,
                            AccessorArray<40,Accessor<MomentQuad,3>,3>(fa_xs),
                            group_start);
        break;
      }
    case 48:
      {
        gpu_expand_scattering_cross_section<48><<<grid,block>>>(
                            subgrid_bounds.lo,
                            AccessorArray<48,Accessor<MomentQuad,2>,2>(fa_slgg),
                            fa_mat,
                            AccessorArray<48,Accessor<MomentQuad,3>,3>(fa_xs),
                            group_start);
        break;
      }
    case 56:
      {
        gpu_expand_scattering_cross_section<56><<<grid,block>>>(
                            subgrid_bounds.lo,
                            AccessorArray<56,Accessor<MomentQuad,2>,2>(fa_slgg),
                            fa_mat,
                            AccessorArray<56,Accessor<MomentQuad,3>,3>(fa_xs),
                            group_start);
        break;
      }
    case 64:
      {
        gpu_expand_scattering_cross_section<64><<<grid,block>>>(
                            subgrid_bounds.lo,
                            AccessorArray<64,Accessor<MomentQuad,2>,2>(fa_slgg),
                            fa_mat,
                            AccessorArray<64,Accessor<MomentQuad,3>,3>(fa_xs),
                            group_start);
        break;
      }
    default:
      assert(false); // add more cases
  }
}

