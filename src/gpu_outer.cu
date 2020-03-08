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

#include "snap.h"
#include "snap_cuda_help.h"

template<int GROUPS, int STRIP_SIZE>
__global__
void gpu_flux0_outer_source(const Point<3> origin,
                            const AccessorArray<GROUPS,
                                    AccessorRO<double,3>,3> fa_qi0,
                            const AccessorArray<GROUPS,
                                    AccessorRO<double,3>,3> fa_flux0,
                            const AccessorArray<GROUPS,
                                    AccessorRO<MomentQuad,2>,2> fa_slgg,
                            const AccessorRO<int,3> fa_mat,
                            const AccessorArray<GROUPS,
                                    AccessorWO<double,3>,3> fa_qo0)
{
  __shared__ double flux_buffer[GROUPS][STRIP_SIZE];
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z;
  const Point<3> p = origin + Point<3>(x,y,z);
  const int group = threadIdx.z;
  const int strip_offset = threadIdx.y * blockDim.x + threadIdx.x;
  // First, update our pointers
  const double *qi0_ptr = fa_qi0[group].ptr(p);
  const double *flux0_ptr = fa_flux0[group].ptr(p);
  const int *mat_ptr = fa_mat.ptr(p);
  double *qo0_ptr = fa_qo0[group].ptr(p);
  // Do a little prefetching of other values we need too
  // Be intelligent about loads, we're trying to keep the slgg
  // matrix in L2 cache so make sure all other loads and stores 
  // are cached with a streaming prefix
  double flux0;
  asm volatile("ld.global.cs.f64 %0, [%1];" : "=d"(flux0) : "l"(flux0_ptr) : "memory");
  // Other threads will use the material so cache at all levels
  int mat;
  asm volatile("ld.global.ca.s32 %0, [%1];" : "=r"(mat) : "l"(mat_ptr) : "memory");
  double qo0;
  asm volatile("ld.global.cs.f64 %0, [%1];" : "=d"(qo0) : "l"(qi0_ptr) : "memory");
  // Write the value into shared
  flux_buffer[group][strip_offset] = flux0;
  // Synchronize when all the writes into shared memory are done
  __syncthreads();
  // Do the math
  #pragma unroll
  for (int g = 0; g < GROUPS; g++) {
    if (g == group)
      continue;
    const MomentQuad *local_slgg = fa_slgg[group].ptr(Point<2>(mat,g));
    double cs;
    asm volatile("ld.global.ca.f64 %0, [%1];" : "=d"(cs) : "l"(local_slgg) : "memory");
    qo0 += cs * flux_buffer[g][strip_offset];
  }
  // Write out our result
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(qo0_ptr), "d"(qo0) : "memory");
}

template<int GROUPS, int MAX_X, int MAX_Y>
__host__
void flux0_launch_helper(Rect<3> subgrid_bounds,
                         const std::vector<AccessorRO<double,3> > fa_qi0,
                         const std::vector<AccessorRO<double,3> > fa_flux0,
                         const std::vector<AccessorRO<MomentQuad,2> > fa_slgg,
                         const AccessorRO<int,3> &fa_mat,
                         const std::vector<AccessorWO<double,3> > fa_qo0)
{
  const int x_range = (subgrid_bounds.hi[0] - subgrid_bounds.lo[0]) + 1;
  const int y_range = (subgrid_bounds.hi[1] - subgrid_bounds.lo[1]) + 1;
  const int z_range = (subgrid_bounds.hi[2] - subgrid_bounds.lo[2]) + 1;
  dim3 block(gcd(x_range,MAX_X), gcd(y_range,MAX_Y), GROUPS);
  dim3 grid(x_range/block.x, y_range/block.y, z_range);
  gpu_flux0_outer_source<GROUPS,MAX_X*MAX_Y><<<grid,block>>>(
                              subgrid_bounds.lo,
                              AccessorArray<GROUPS,
                                AccessorRO<double,3>,3>(fa_qi0),
                              AccessorArray<GROUPS,
                                AccessorRO<double,3>,3>(fa_flux0),
                              AccessorArray<GROUPS,
                                AccessorRO<MomentQuad,2>,2>(fa_slgg),
                              fa_mat,
                              AccessorArray<GROUPS,
                                AccessorWO<double,3>,3>(fa_qo0));
}

__host__
void run_flux0_outer_source(Rect<3> subgrid_bounds,
                         const std::vector<AccessorRO<double,3> > &fa_qi0,
                         const std::vector<AccessorRO<double,3> > &fa_flux0,
                         const std::vector<AccessorRO<MomentQuad,2> > &fa_slgg,
                         const std::vector<AccessorWO<double,3> > &fa_qo0,
                         const AccessorRO<int,3> &fa_mat, const int num_groups)
{
  // TODO: replace this template madness with Terra
#define GROUP_CASE(g,x,y)                                                           \
  case g:                                                                           \
    {                                                                               \
      flux0_launch_helper<g,x,y>(subgrid_bounds, fa_qi0, fa_flux0, fa_slgg,         \
                                 fa_mat, fa_qo0);                                   \
      break;                                                                        \
    }
  switch (num_groups)
  {
    GROUP_CASE(1,32,32)
    GROUP_CASE(2,32,16)
    GROUP_CASE(3,32,8)
    GROUP_CASE(4,32,8)
    GROUP_CASE(5,32,4)
    GROUP_CASE(6,32,4)
    GROUP_CASE(7,32,4)
    GROUP_CASE(8,32,4)
    GROUP_CASE(9,32,2)
    GROUP_CASE(10,32,2)
    GROUP_CASE(11,32,2)
    GROUP_CASE(12,32,2)
    GROUP_CASE(13,32,2)
    GROUP_CASE(14,32,2)
    GROUP_CASE(15,32,2)
    GROUP_CASE(16,32,2)
    GROUP_CASE(17,16,2)
    GROUP_CASE(18,16,2)
    GROUP_CASE(19,16,2)
    GROUP_CASE(20,16,2)
    GROUP_CASE(21,16,2)
    GROUP_CASE(22,16,2)
    GROUP_CASE(23,16,2)
    GROUP_CASE(24,16,2)
    GROUP_CASE(25,16,2)
    GROUP_CASE(26,16,2)
    GROUP_CASE(27,16,2)
    GROUP_CASE(28,16,2)
    GROUP_CASE(29,16,2)
    GROUP_CASE(30,16,2)
    GROUP_CASE(31,16,2)
    GROUP_CASE(32,16,2)
    GROUP_CASE(33,16,1)
    GROUP_CASE(34,16,1)
    GROUP_CASE(35,16,1)
    GROUP_CASE(36,16,1)
    GROUP_CASE(37,16,1)
    GROUP_CASE(38,16,1)
    GROUP_CASE(39,16,1)
    GROUP_CASE(40,16,1)
    GROUP_CASE(41,16,1)
    GROUP_CASE(42,16,1)
    GROUP_CASE(43,16,1)
    GROUP_CASE(44,16,1)
    GROUP_CASE(45,16,1)
    GROUP_CASE(46,16,1)
    GROUP_CASE(47,16,1)
    GROUP_CASE(48,16,1)
    GROUP_CASE(49,16,1)
    GROUP_CASE(50,16,1)
    GROUP_CASE(51,16,1)
    GROUP_CASE(52,16,1)
    GROUP_CASE(53,16,1)
    GROUP_CASE(54,16,1)
    GROUP_CASE(55,16,1)
    GROUP_CASE(56,16,1)
    GROUP_CASE(57,16,1)
    GROUP_CASE(58,16,1)
    GROUP_CASE(59,16,1)
    GROUP_CASE(60,16,1)
    GROUP_CASE(61,16,1)
    GROUP_CASE(62,16,1)
    GROUP_CASE(63,16,1)
    GROUP_CASE(64,16,1)
    GROUP_CASE(65,8,1)
    GROUP_CASE(66,8,1)
    GROUP_CASE(67,8,1)
    GROUP_CASE(68,8,1)
    GROUP_CASE(69,8,1)
    GROUP_CASE(70,8,1)
    GROUP_CASE(71,8,1)
    GROUP_CASE(72,8,1)
    GROUP_CASE(73,8,1)
    GROUP_CASE(74,8,1)
    GROUP_CASE(75,8,1)
    GROUP_CASE(76,8,1)
    GROUP_CASE(77,8,1)
    GROUP_CASE(78,8,1)
    GROUP_CASE(79,8,1)
    GROUP_CASE(80,8,1)
    GROUP_CASE(81,8,1)
    GROUP_CASE(82,8,1)
    GROUP_CASE(83,8,1)
    GROUP_CASE(84,8,1)
    GROUP_CASE(85,8,1)
    GROUP_CASE(86,8,1)
    GROUP_CASE(87,8,1)
    GROUP_CASE(88,8,1)
    GROUP_CASE(89,8,1)
    GROUP_CASE(90,8,1)
    GROUP_CASE(91,8,1)
    GROUP_CASE(92,8,1)
    GROUP_CASE(93,8,1)
    GROUP_CASE(94,8,1)
    GROUP_CASE(95,8,1)
    GROUP_CASE(96,8,1)
    // About to drop down to 1 CTA per SM due to shared memory
    default:
      printf("Adding group case to outer flux0 computation!\n");
      assert(false);
  }
#undef GROUP_CASE
}

template<int GROUPS, int STRIP_SIZE>
__global__
void gpu_fluxm_outer_source(const Point<3> origin,
                            const AccessorArray<GROUPS,
                                    AccessorRO<MomentTriple,3>,3> fa_fluxm,
                            const AccessorArray<GROUPS,
                                    AccessorRO<MomentQuad,2>,2> fa_slgg,
                            const AccessorRO<int,3> fa_mat,
                            const AccessorArray<GROUPS,
                                    AccessorWO<MomentTriple,3>,3> fa_qom,
                            const int           num_moments,
                            const ConstBuffer<4,int> lma)
{
  __shared__ double fluxm_buffer_0[GROUPS][STRIP_SIZE];
  __shared__ double fluxm_buffer_1[GROUPS][STRIP_SIZE];
  __shared__ double fluxm_buffer_2[GROUPS][STRIP_SIZE];
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y + blockDim.y + threadIdx.y;
  const int z = blockIdx.z;
  const Point<3> p = origin + Point<3>(x,y,z);
  const int group = threadIdx.z;
  const int strip_offset = threadIdx.y * blockDim.x + threadIdx.x;
  const MomentTriple *fluxm_ptr = fa_fluxm[group].ptr(p);
  const int *mat_ptr = fa_mat.ptr(p);
  MomentTriple *qom_ptr = fa_qom[group].ptr(p);
  MomentTriple fluxm;
  asm volatile("ld.global.cs.v2.f64 {%0,%1}, [%2];" : "=d"(fluxm[0]), "=d"(fluxm[1]) 
                : "l"(fluxm_ptr) : "memory");
  asm volatile("ld.global.cs.f64 %0, [%1];" : "=d"(fluxm[2]) 
                : "l"(((char*)fluxm_ptr)+16) : "memory");
  int mat;
  asm volatile("ld.global.ca.s32 %0, [%1];" : "=r"(mat) : "l"(mat_ptr) : "memory");
  // Write the fluxm into shared memory
  fluxm_buffer_0[group][strip_offset] = fluxm[0];
  fluxm_buffer_1[group][strip_offset] = fluxm[1];
  fluxm_buffer_2[group][strip_offset] = fluxm[2];
  // Synchronize to make sure all the writes to shared are done 
  __syncthreads();
  // Do the math
  MomentTriple qom;
  #pragma unroll
  for (int g = 0; g < GROUPS; g++) {
    if (g == group)
      continue;
    int moment = 0;
    const MomentQuad *local_slgg = fa_slgg[group].ptr(Point<2>(mat, g));
    MomentQuad scat;
    asm volatile("ld.global.ca.v2.f64 {%0,%1}, [%2];" : "=d"(scat[0]), "=d"(scat[1])
                  : "l"(local_slgg) : "memory");
    asm volatile("ld.global.ca.v2.f64 {%0,%1}, [%2];" : "=d"(scat[2]), "=d"(scat[3])
                  : "l"(((char*)local_slgg)+16) : "memory");
    MomentTriple csm;
    for (int l = 1; l < num_moments; l++) {
      for (int j = 0; j < lma[l]; j++)
        csm[moment+j] = scat[l];
      moment += lma[l];
    }
    fluxm[0] = fluxm_buffer_0[g][strip_offset];
    fluxm[1] = fluxm_buffer_1[g][strip_offset];
    fluxm[2] = fluxm_buffer_2[g][strip_offset];
    for (int l = 0; l < (num_moments-1); l++)
      qom[l] += csm[l] * fluxm[l];
  }
  // Now we can write out the result
  asm volatile("st.global.cs.v2.f64 [%0], {%1,%2};" : : "l"(qom_ptr), 
                "d"(qom[0]), "d"(qom[1]) : "memory");
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(((char*)qom_ptr)+16),
                "d"(qom[2]) : "memory");
}

template<int GROUPS, int MAX_X, int MAX_Y>
__host__
void fluxm_launch_helper(Rect<3> subgrid_bounds,
                         const std::vector<AccessorRO<MomentTriple,3> > &fa_fluxm,
                         const std::vector<AccessorRO<MomentQuad,2> > &fa_slgg,
                         const std::vector<AccessorWO<MomentTriple,3> > &fa_qom,
                         const AccessorRO<int,3> &fa_mat,
                         const int num_groups, const int num_moments, const int lma[4])
{
  const int x_range = (subgrid_bounds.hi[0] - subgrid_bounds.lo[0]) + 1;
  const int y_range = (subgrid_bounds.hi[1] - subgrid_bounds.lo[1]) + 1;
  const int z_range = (subgrid_bounds.hi[2] - subgrid_bounds.lo[2]) + 1;
  dim3 block(gcd(x_range,MAX_X), gcd(y_range,MAX_Y), GROUPS);
  dim3 grid(x_range/block.x, y_range/block.y, z_range);
  gpu_fluxm_outer_source<GROUPS,MAX_X*MAX_Y><<<grid,block>>>(
                            subgrid_bounds.lo,
                            AccessorArray<GROUPS,
                              AccessorRO<MomentTriple,3>,3>(fa_fluxm),
                            AccessorArray<GROUPS,
                              AccessorRO<MomentQuad,2>,2>(fa_slgg), fa_mat,
                            AccessorArray<GROUPS,
                              AccessorWO<MomentTriple,3>,3>(fa_qom),
                            num_moments, ConstBuffer<4,int>(lma));
}

__host__
void run_fluxm_outer_source(Rect<3> subgrid_bounds,
                         const std::vector<AccessorRO<MomentTriple,3> > &fa_fluxm,
                         const std::vector<AccessorRO<MomentQuad,2> > &fa_slgg,
                         const std::vector<AccessorWO<MomentTriple,3> > &fa_qom,
                         const AccessorRO<int,3> &fa_mat,
                         const int num_groups, const int num_moments, const int lma[4])
{
  // TODO: replace this template madness with Terra
#define GROUP_CASE(g,x,y)                                                         \
  case g:                                                                         \
    {                                                                             \
      fluxm_launch_helper<g,x,y>(subgrid_bounds, fa_fluxm, fa_slgg, fa_qom,       \
                                 fa_mat,  num_groups, num_moments, lma);          \
      break;                                                                      \
    }
  switch (num_groups)
  {
    GROUP_CASE(1,32,32)
    GROUP_CASE(2,32,16)
    GROUP_CASE(3,32,8)
    GROUP_CASE(4,32,8)
    GROUP_CASE(5,32,4)
    GROUP_CASE(6,32,4)
    GROUP_CASE(7,32,4)
    GROUP_CASE(8,32,4)
    GROUP_CASE(9,32,2)
    GROUP_CASE(10,32,2)
    GROUP_CASE(11,32,2)
    GROUP_CASE(12,32,2)
    GROUP_CASE(13,32,2)
    GROUP_CASE(14,32,2)
    GROUP_CASE(15,32,2)
    GROUP_CASE(16,32,2)
#if 0
    GROUP_CASE(17,16,2)
    GROUP_CASE(18,16,2)
    GROUP_CASE(19,16,2)
    GROUP_CASE(20,16,2)
    GROUP_CASE(21,16,2)
    GROUP_CASE(22,16,2)
    GROUP_CASE(23,16,2)
    GROUP_CASE(24,16,2)
    GROUP_CASE(25,16,2)
    GROUP_CASE(26,16,2)
    GROUP_CASE(27,16,2)
    GROUP_CASE(28,16,2)
    GROUP_CASE(29,16,2)
    GROUP_CASE(30,16,2)
    GROUP_CASE(31,16,2)
    GROUP_CASE(32,16,2)
    GROUP_CASE(33,16,1)
    GROUP_CASE(34,16,1)
    GROUP_CASE(35,16,1)
    GROUP_CASE(36,16,1)
    GROUP_CASE(37,16,1)
    GROUP_CASE(38,16,1)
    GROUP_CASE(39,16,1)
    GROUP_CASE(40,16,1)
    GROUP_CASE(41,16,1)
    GROUP_CASE(42,16,1)
    GROUP_CASE(43,16,1)
    GROUP_CASE(44,16,1)
    GROUP_CASE(45,16,1)
    GROUP_CASE(46,16,1)
    GROUP_CASE(47,16,1)
    GROUP_CASE(48,16,1)
    GROUP_CASE(49,16,1)
    GROUP_CASE(50,16,1)
    GROUP_CASE(51,16,1)
    GROUP_CASE(52,16,1)
    GROUP_CASE(53,16,1)
    GROUP_CASE(54,16,1)
    GROUP_CASE(55,16,1)
    GROUP_CASE(56,16,1)
    GROUP_CASE(57,16,1)
    GROUP_CASE(58,16,1)
    GROUP_CASE(59,16,1)
    GROUP_CASE(60,16,1)
    GROUP_CASE(61,16,1)
    GROUP_CASE(62,16,1)
    GROUP_CASE(63,16,1)
    GROUP_CASE(64,16,1)
    GROUP_CASE(65,8,1)
    GROUP_CASE(66,8,1)
    GROUP_CASE(67,8,1)
    GROUP_CASE(68,8,1)
    GROUP_CASE(69,8,1)
    GROUP_CASE(70,8,1)
    GROUP_CASE(71,8,1)
    GROUP_CASE(72,8,1)
    GROUP_CASE(73,8,1)
    GROUP_CASE(74,8,1)
    GROUP_CASE(75,8,1)
    GROUP_CASE(76,8,1)
    GROUP_CASE(77,8,1)
    GROUP_CASE(78,8,1)
    GROUP_CASE(79,8,1)
    GROUP_CASE(80,8,1)
    GROUP_CASE(81,8,1)
    GROUP_CASE(82,8,1)
    GROUP_CASE(83,8,1)
    GROUP_CASE(84,8,1)
    GROUP_CASE(85,8,1)
    GROUP_CASE(86,8,1)
    GROUP_CASE(87,8,1)
    GROUP_CASE(88,8,1)
    GROUP_CASE(89,8,1)
    GROUP_CASE(90,8,1)
    GROUP_CASE(91,8,1)
    GROUP_CASE(92,8,1)
    GROUP_CASE(93,8,1)
    GROUP_CASE(94,8,1)
    GROUP_CASE(95,8,1)
    GROUP_CASE(96,8,1)
#endif
    default:
      printf("Adding group case to outer fluxm computation!\n");
      assert(false);
  }
#undef GROUP_CASE
}

__global__
void gpu_outer_convergence(const Point<3> origin,
                           const AccessorRO<double,3> fa_flux0,
                           const AccessorRO<double,3> fa_flux0po,
                           const double epsi, 
                           const DeferredBuffer<int,1> results,
                           const int results_offset)
{
  // We know there is never more than 32 warps in a CTA
  __shared__ int trampoline[32];

  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;
  const Point<3> p = origin + Point<3>(x,y,z);

  const double *flux0_ptr = fa_flux0.ptr(p);
  const double *flux0po_ptr = fa_flux0po.ptr(p);

  const double tolr = 1.0e-12;
  
  double flux0po = *flux0po_ptr;
  double df = 1.0;
  if (fabs(flux0po) < tolr) {
    flux0po = 1.0;
    df = 0.0;
  }
  double flux0 = *flux0_ptr;
  df = fabs( (flux0 / flux0po) - df );
  int local_converged = 1;
  if ((df >= -INFINITY) && (df > epsi))
    local_converged = 0;
  // Perform a local reduction inside the CTA
  // Butterfly reduction across all threads in all warps
  for (int i = 16; i >= 1; i/=2)
    local_converged += __shfl_xor_sync(0xfffffff, local_converged, i, 32);
  unsigned laneid;
  asm volatile("mov.u32 %0, %laneid;" : "=r"(laneid) : );
  unsigned warpid = 
    ((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x) >> 5;
  // First thread in each warp writes out all values
  if (laneid == 0)
    trampoline[warpid] = local_converged;
  __syncthreads();
  // Butterfly reduction across all thread in the first warp
  if (warpid == 0) {
    unsigned numwarps = (blockDim.x * blockDim.y * blockDim.z) >> 5;
    local_converged = (laneid < numwarps) ? trampoline[laneid] : 0;
    for (int i = 16; i >= 1; i/=2)
      local_converged += __shfl_xor_sync(0xfffffff, local_converged, i, 32);
    // First thread does the atomic
    if (laneid == 0)
      results.write(Point<1>(results_offset + 
        (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x), local_converged);
  }
}

__global__
void gpu_sum_outer_convergence(const DeferredBuffer<int,1> buffer,
                               const DeferredValue<bool> result,
                               const size_t total_blocks, 
                               const int expected)
{
  __shared__ int trampoline[32];
  int offset = threadIdx.x;
  int total = 0;
  while (offset < total_blocks) {
    total += buffer.read(Point<1>(offset));
    offset += blockDim.x;
  }
  for (int i = 16; i >= 1; i/=2)
    total += __shfl_xor_sync(0xfffffff, total, i, 32);
  unsigned laneid;
  asm volatile("mov.u32 %0, %laneid;" : "=r"(laneid) : );
  unsigned warpid = threadIdx.x >> 5;
  // Write results in the trampoline
  if (laneid == 0)
    trampoline[warpid] = total;
  __syncthreads();
  if (warpid == 0)
  {
    unsigned numwarps = blockDim.x >> 5;
    total = (laneid < numwarps) ? trampoline[laneid] : 0;
    for (int i = 16; i >= 1; i/=2)
      total += __shfl_xor_sync(0xfffffff, total, i, 32);
    if (laneid == 0)
      result.write(total == expected);
  }
}

__host__
void run_outer_convergence(Rect<3> subgrid_bounds,
                           const DeferredValue<bool> &result,
                           const std::vector<AccessorRO<double,3> > &fa_flux0,
                           const std::vector<AccessorRO<double,3> > &fa_flux0po,
                           const double epsi)
{
  
  // Launch the kernels
  const int x_range = (subgrid_bounds.hi[0] - subgrid_bounds.lo[0]) + 1;
  const int y_range = (subgrid_bounds.hi[1] - subgrid_bounds.lo[1]) + 1;
  const int z_range = (subgrid_bounds.hi[2] - subgrid_bounds.lo[2]) + 1;
  dim3 block(gcd(x_range,32),gcd(y_range,4),gcd(z_range,4));
  dim3 grid(x_range/block.x, y_range/block.y, z_range/block.z);
  const size_t total_blocks = grid.x*grid.y*grid.z;
  assert(fa_flux0.size() == fa_flux0po.size());
  const Rect<1> bounds(Point<1>(0),Point<1>(total_blocks * fa_flux0.size() - 1));
  DeferredBuffer<int,1> buffer(bounds, Memory::GPU_FB_MEM);
  for (unsigned idx = 0; idx < fa_flux0.size(); idx++) {
    gpu_outer_convergence<<<grid,block>>>(subgrid_bounds.lo,
                                          fa_flux0[idx], fa_flux0po[idx],
                                          epsi, buffer, idx * total_blocks);
  }
  dim3 block2((bounds.hi[0]+1) > 1024 ? 1024 : (bounds.hi[0]+1),1,1);
  // Round up to the nearest multiple of warps
  while ((block2.x % 32) != 0)
    block2.x++;
  dim3 grid2(1,1,1);
  const int expected = x_range * y_range * z_range * fa_flux0.size();
  gpu_sum_outer_convergence<<<grid2,block2>>>(buffer, result, bounds.hi[0]+1, expected);
}

