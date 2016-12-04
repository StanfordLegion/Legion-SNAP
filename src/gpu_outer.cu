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

template<int GROUPS, int STRIP_SIZE>
__global__
void gpu_flux0_outer_source(const double      *qi0_ptrs[GROUPS], 
                            const double      *flux0_ptrs[GROUPS],
                            const MomentQuad  *slgg_ptrs[GROUPS],
                            const int         *mat_ptr,
                                  double      *qo0_ptrs[GROUPS],
                            ByteOffset        qi0_offsets[3],
                            ByteOffset        flux0_offsets[3],
                            ByteOffset        slgg_offsets[2],
                            ByteOffset        mat_offsets[3],
                            ByteOffset        qo0_offsets[3],
                            const int         x_start,
                            const int         y_start,
                            const int         z_start)
{
  __shared__ double flux_buffer[GROUPS][STRIP_SIZE];
  const int x = blockIdx.x * STRIP_SIZE + threadIdx.x + x_start;
  const int y = blockIdx.y + y_start;
  const int z = blockIdx.z + z_start;
  const int group = threadIdx.y;
  // First, update our pointers
  const double *qi0_ptr = qi0_ptrs[group] + x * qi0_offsets[0] +
    y * qi0_offsets[1] + z * qi0_offsets[2];
  const double *flux0_ptr = flux0_ptrs[group] + x * flux0_offsets[0] +
    y * flux0_offsets[1] + z * flux0_offsets[2];
  mat_ptr += x * mat_offsets[0] + y * mat_offsets[1] + z * mat_offsets[2];
  double *qo0_ptr = qo0_ptrs[group] + x * qo0_offsets[0] + 
    y *qo0_offsets[1] + z * qo0_offsets[2];
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
  flux_buffer[group][x] = flux0;
  // Can compute our slgg_ptr with the matrix result
  const MomentQuad *slgg_ptr = slgg_ptrs[group] + mat * slgg_offsets[0];
  // Synchronize when all the writes into shared memory are done
  __syncthreads();
  // Do the math
  #pragma unroll
  for (int g = 0; g < GROUPS; g++) {
    if (g == group)
      continue;
    const MomentQuad *local_slgg = slgg_ptr + g * slgg_offsets[1];
    double cs;
    asm volatile("ld.global.ca.f64 %0, [%1];" : "=d"(cs) : "l"(local_slgg) : "memory");
    qo0 += cs * flux_buffer[g][x];
  }
  // Write out our result
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(qo0_ptr), "d"(qo0) : "memory");
}

template<int GROUPS, int MAX_THREADS>
__host__
void flux0_launch_helper(Rect<3> subgrid_bounds,
                         std::vector<double*> &qi0_vector,
                         std::vector<double*> &flux0_vector,
                         std::vector<MomentQuad*> &slgg_vector,
                         std::vector<double*> &qo0_vector, int *mat_ptr, 
                         ByteOffset qi0_offsets[3], ByteOffset flux0_offsets[3],
                         ByteOffset slgg_offsets[2], ByteOffset qo0_offsets[3],
                         ByteOffset mat_offsets[3])
{
  // Pack the ptrs
  const double *qi0_ptrs[GROUPS];
  const double *flux0_ptrs[GROUPS];
  const MomentQuad *slgg_ptrs[GROUPS];
  double *qo0_ptrs[GROUPS];
  for (int i = 0; i < GROUPS; i++) {
    qi0_ptrs[i] = qi0_vector[i];
    flux0_ptrs[i] = flux0_vector[i];
    slgg_ptrs[i] = slgg_vector[i];
    qo0_ptrs[i] = qo0_vector[i];
  }
  const int x_start = subgrid_bounds.lo[0];
  const int y_start = subgrid_bounds.lo[1];
  const int z_start = subgrid_bounds.lo[2];
  const int x_range = (subgrid_bounds.hi[0] - subgrid_bounds.lo[0]) + 1;
  const int y_range = (subgrid_bounds.hi[1] - subgrid_bounds.lo[1]) + 1;
  const int z_range = (subgrid_bounds.hi[2] - subgrid_bounds.lo[2]) + 1;
  assert((x_range % 32) == 0);
  dim3 block(32, GROUPS, 1);
  dim3 grid(x_range/32, y_range, z_range);
  gpu_flux0_outer_source<GROUPS,32><<<grid,block>>>(qi0_ptrs, flux0_ptrs,
                                                    slgg_ptrs, mat_ptr,
                                                    qo0_ptrs, qi0_offsets,
                                                    flux0_offsets, slgg_offsets,
                                                    mat_offsets, qo0_offsets,
                                                    x_start, y_start, z_start);
}

__host__
void run_flux0_outer_source(Rect<3> subgrid_bounds,
                            std::vector<double*> &qi0_ptrs,
                            std::vector<double*> &flux0_ptrs,
                            std::vector<MomentQuad*> &slgg_ptrs,
                            std::vector<double*> &qo0_ptrs, int *mat_ptr, 
                            ByteOffset qi0_offsets[3], ByteOffset flux0_offsets[3],
                            ByteOffset slgg_offsets[2], ByteOffset qo0_offsets[3],
                            ByteOffset mat_offsets[3], const int num_groups)
{
  // TODO: replace this template madness with Terra
#define GROUP_CASE(g,t)                                                             \
  case g:                                                                           \
    {                                                                               \
      flux0_launch_helper<g,t>(subgrid_bounds, qi0_ptrs, flux0_ptrs, slgg_ptrs,     \
                               qo0_ptrs, mat_ptr, qi0_offsets, flux0_offsets,       \
                               slgg_offsets, qo0_offsets, mat_offsets);             \
      break;                                                                        \
    }
  switch (num_groups)
  {
    GROUP_CASE(1,256)
    GROUP_CASE(2,256)
    GROUP_CASE(3,256)
    GROUP_CASE(4,256)
    GROUP_CASE(5,256)
    GROUP_CASE(6,256)
    GROUP_CASE(7,256)
    GROUP_CASE(8,256)
    GROUP_CASE(9,256)
    GROUP_CASE(10,256)
    GROUP_CASE(11,256)
    GROUP_CASE(12,256)
    GROUP_CASE(13,192)
    GROUP_CASE(14,192)
    GROUP_CASE(15,192)
    GROUP_CASE(16,192)
    GROUP_CASE(17,128)
    GROUP_CASE(18,128)
    GROUP_CASE(19,128)
    GROUP_CASE(20,128)
    GROUP_CASE(21,128)
    GROUP_CASE(22,128)
    GROUP_CASE(23,128)
    GROUP_CASE(24,128)
    GROUP_CASE(25,96)
    GROUP_CASE(26,96)
    GROUP_CASE(27,96)
    GROUP_CASE(28,96)
    GROUP_CASE(29,96)
    GROUP_CASE(30,96)
    GROUP_CASE(31,96)
    GROUP_CASE(32,96)
    GROUP_CASE(33,64)
    GROUP_CASE(34,64)
    GROUP_CASE(35,64)
    GROUP_CASE(36,64)
    GROUP_CASE(37,64)
    GROUP_CASE(38,64)
    GROUP_CASE(39,64)
    GROUP_CASE(40,64)
    GROUP_CASE(41,64)
    GROUP_CASE(42,64)
    GROUP_CASE(43,64)
    GROUP_CASE(44,64)
    GROUP_CASE(45,64)
    GROUP_CASE(46,64)
    GROUP_CASE(47,64)
    GROUP_CASE(48,64)
    GROUP_CASE(49,32)
    GROUP_CASE(50,32)
    GROUP_CASE(51,32)
    GROUP_CASE(52,32)
    GROUP_CASE(53,32)
    GROUP_CASE(54,32)
    GROUP_CASE(55,32)
    GROUP_CASE(56,32)
    GROUP_CASE(57,32)
    GROUP_CASE(58,32)
    GROUP_CASE(59,32)
    GROUP_CASE(60,32)
    GROUP_CASE(61,32)
    GROUP_CASE(62,32)
    GROUP_CASE(63,32)
    GROUP_CASE(64,32)
    GROUP_CASE(65,32)
    GROUP_CASE(66,32)
    GROUP_CASE(67,32)
    GROUP_CASE(68,32)
    GROUP_CASE(69,32)
    GROUP_CASE(70,32)
    GROUP_CASE(71,32)
    GROUP_CASE(72,32)
    GROUP_CASE(73,32)
    GROUP_CASE(74,32)
    GROUP_CASE(75,32)
    GROUP_CASE(76,32)
    GROUP_CASE(77,32)
    GROUP_CASE(78,32)
    GROUP_CASE(79,32)
    GROUP_CASE(80,32)
    GROUP_CASE(81,32)
    GROUP_CASE(82,32)
    GROUP_CASE(83,32)
    GROUP_CASE(84,32)
    GROUP_CASE(85,32)
    GROUP_CASE(86,32)
    GROUP_CASE(87,32)
    GROUP_CASE(88,32)
    GROUP_CASE(89,32)
    GROUP_CASE(90,32)
    GROUP_CASE(91,32)
    GROUP_CASE(92,32)
    GROUP_CASE(93,32)
    GROUP_CASE(94,32)
    GROUP_CASE(95,32)
    GROUP_CASE(96,32)
    // About to drop down to 1 CTA per SM due to shared memory
    default:
      printf("Adding group case to outer flux0 computation!\n");
      assert(false);
  }
#undef GROUP_CASE
}

template<int GROUPS, int STRIP_SIZE>
__global__
void gpu_fluxm_outer_source(const MomentTriple  *fluxm_ptrs[GROUPS],
                            const MomentQuad    *slgg_ptrs[GROUPS],
                            const int           *mat_ptr,
                                  MomentTriple  *qom_ptrs[GROUPS],
                            ByteOffset          fluxm_offsets[3],
                            ByteOffset          slgg_offsets[2],
                            ByteOffset          mat_offsets[3],
                            ByteOffset          qom_offsets[3],
                            const int           num_moments,
                            const int           lma[4],
                            const int           x_start,
                            const int           y_start,
                            const int           z_start)
{
  __shared__ double fluxm_buffer_0[GROUPS][STRIP_SIZE];
  __shared__ double fluxm_buffer_1[GROUPS][STRIP_SIZE];
  __shared__ double fluxm_buffer_2[GROUPS][STRIP_SIZE];
  const int x = blockIdx.x * STRIP_SIZE + threadIdx.x + x_start;
  const int y = blockIdx.y + y_start;
  const int z = blockIdx.z + z_start;
  const int group = threadIdx.y;
  const MomentTriple *fluxm_ptr = fluxm_ptrs[group] + x * fluxm_offsets[0] +
    y * fluxm_offsets[1] + z * fluxm_offsets[2];
  mat_ptr += x * mat_offsets[0] + y * mat_offsets[1] + z * mat_offsets[2];
  MomentTriple *qom_ptr = qom_ptrs[group] + x * qom_offsets[0] + 
    y *qom_offsets[1] + z * qom_offsets[2];
  MomentTriple fluxm;
  asm volatile("ld.global.cs.v2.f64 {%0,%1}, [%2];" : "=d"(fluxm[0]), "=d"(fluxm[1]) 
                : "l"(fluxm_ptr) : "memory");
  asm volatile("ld.global.cs.f64 %0, [%1];" : "=d"(fluxm[2]) 
                : "l"(((char*)fluxm_ptr)+16) : "memory");
  int mat;
  asm volatile("ld.global.ca.s32 %0, [%1];" : "=r"(mat) : "l"(mat_ptr) : "memory");
  // Write the fluxm into shared memory
  fluxm_buffer_0[group][x] = fluxm[0];
  fluxm_buffer_1[group][x] = fluxm[1];
  fluxm_buffer_2[group][x] = fluxm[2];
  // Can compute our slgg_ptr with the matrix result
  const MomentQuad *slgg_ptr = slgg_ptrs[group] + mat * slgg_offsets[0];
  // Synchronize to make sure all the writes to shared are done 
  __syncthreads();
  // Do the math
  MomentTriple qom;
  #pragma unroll
  for (int g = 0; g < GROUPS; g++) {
    if (g == group)
      continue;
    int moment = 0;
    const MomentQuad *local_slgg = slgg_ptr + g * slgg_offsets[1];
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
    fluxm[0] = fluxm_buffer_0[g][x];
    fluxm[1] = fluxm_buffer_1[g][x];
    fluxm[2] = fluxm_buffer_2[g][x];
    for (int l = 0; l < (num_moments-1); l++)
      qom[l] += csm[l] * fluxm[l];
  }
  // Now we can write out the result
  asm volatile("st.global.cs.v2.f64 [%0], {%1,%2};" : : "l"(qom_ptr), 
                "d"(qom[0]), "d"(qom[1]) : "memory");
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(((char*)qom_ptr)+16),
                "d"(qom[2]) : "memory");
}

template<int GROUPS>
__host__
void fluxm_launch_helper(Rect<3> subgrid_bounds,
                         std::vector<MomentTriple*> &fluxm_vector,
                         std::vector<MomentQuad*> &slgg_vector,
                         std::vector<MomentTriple*> &qom_vector, int *mat_ptr, 
                         ByteOffset fluxm_offsets[3], ByteOffset slgg_offsets[2],
                         ByteOffset mat_offsets[3], ByteOffset qom_offsets[3],
                         const int num_groups, const int num_moments, const int lma[4])
{
  // Pack the ptrs
  const MomentTriple *fluxm_ptrs[GROUPS];
  const MomentQuad *slgg_ptrs[GROUPS];
  MomentTriple *qom_ptrs[GROUPS];
  for (int i = 0; i < GROUPS; i++) {
    fluxm_ptrs[i] = fluxm_vector[i];
    slgg_ptrs[i] = slgg_vector[i];
    qom_ptrs[i] = qom_vector[i];
  }
  const int x_start = subgrid_bounds.lo[0];
  const int y_start = subgrid_bounds.lo[1];
  const int z_start = subgrid_bounds.lo[2];
  const int x_range = (subgrid_bounds.hi[0] - subgrid_bounds.lo[0]) + 1;
  const int y_range = (subgrid_bounds.hi[1] - subgrid_bounds.lo[1]) + 1;
  const int z_range = (subgrid_bounds.hi[2] - subgrid_bounds.lo[2]) + 1;
  assert((x_range % 32) == 0);
  dim3 block(32, GROUPS, 1);
  dim3 grid(x_range/32, y_range, z_range);
  gpu_fluxm_outer_source<GROUPS,32><<<grid,block>>>(fluxm_ptrs, slgg_ptrs,
                                                    mat_ptr, qom_ptrs, 
                                                    fluxm_offsets, slgg_offsets,
                                                    mat_offsets, qom_offsets,
                                                    num_moments, lma, x_start,
                                                    y_start, z_start);
}

__host__
void run_fluxm_outer_source(Rect<3> subgrid_bounds,
                            std::vector<MomentTriple*> &fluxm_ptrs,
                            std::vector<MomentQuad*> &slgg_ptrs,
                            std::vector<MomentTriple*> &qom_ptrs, int *mat_ptr, 
                            ByteOffset fluxm_offsets[3], ByteOffset slgg_offsets[2],
                            ByteOffset mat_offsets[3], ByteOffset qom_offsets[3],
                            const int num_groups, const int num_moments, const int lma[4])
{
  // TODO: replace this template madness with Terra
#define GROUP_CASE(g)                                                             \
  case g:                                                                         \
    {                                                                             \
      fluxm_launch_helper<g>(subgrid_bounds, fluxm_ptrs, slgg_ptrs, qom_ptrs,     \
                             mat_ptr, fluxm_offsets, slgg_offsets, mat_offsets,   \
                             qom_offsets, num_groups, num_moments, lma);          \
      break;                                                                      \
    }
  switch (num_groups)
  {
    GROUP_CASE(1)
    GROUP_CASE(2)
    GROUP_CASE(3)
    GROUP_CASE(4)
    GROUP_CASE(5)
    GROUP_CASE(6)
    GROUP_CASE(7)
    GROUP_CASE(8)
    GROUP_CASE(9)
    GROUP_CASE(10)
    GROUP_CASE(11)
    GROUP_CASE(12)
    GROUP_CASE(13)
    GROUP_CASE(14)
    GROUP_CASE(15)
    GROUP_CASE(16)
    GROUP_CASE(17)
    GROUP_CASE(18)
    GROUP_CASE(19)
    GROUP_CASE(20)
    GROUP_CASE(21)
    GROUP_CASE(22)
    GROUP_CASE(23)
    GROUP_CASE(24)
    GROUP_CASE(25)
    GROUP_CASE(26)
    GROUP_CASE(27)
    GROUP_CASE(28)
    GROUP_CASE(29)
    GROUP_CASE(30)
    GROUP_CASE(31)
    GROUP_CASE(32)
    GROUP_CASE(33)
    GROUP_CASE(34)
    GROUP_CASE(35)
    GROUP_CASE(36)
    GROUP_CASE(37)
    GROUP_CASE(38)
    GROUP_CASE(39)
    GROUP_CASE(40)
    GROUP_CASE(41)
    GROUP_CASE(42)
    GROUP_CASE(43)
    GROUP_CASE(44)
    GROUP_CASE(45)
    GROUP_CASE(46)
    GROUP_CASE(47)
    GROUP_CASE(48)
    GROUP_CASE(49)
    GROUP_CASE(50)
    GROUP_CASE(51)
    GROUP_CASE(52)
    GROUP_CASE(53)
    GROUP_CASE(54)
    GROUP_CASE(55)
    GROUP_CASE(56)
    GROUP_CASE(57)
    GROUP_CASE(58)
    GROUP_CASE(59)
    GROUP_CASE(60)
    GROUP_CASE(61)
    GROUP_CASE(62)
    GROUP_CASE(63)
    GROUP_CASE(64)
#if 0
    GROUP_CASE(65)
    GROUP_CASE(66)
    GROUP_CASE(67)
    GROUP_CASE(68)
    GROUP_CASE(69)
    GROUP_CASE(70)
    GROUP_CASE(71)
    GROUP_CASE(72)
    GROUP_CASE(73)
    GROUP_CASE(74)
    GROUP_CASE(75)
    GROUP_CASE(76)
    GROUP_CASE(77)
    GROUP_CASE(78)
    GROUP_CASE(79)
    GROUP_CASE(80)
    GROUP_CASE(81)
    GROUP_CASE(82)
    GROUP_CASE(83)
    GROUP_CASE(84)
    GROUP_CASE(85)
    GROUP_CASE(86)
    GROUP_CASE(87)
    GROUP_CASE(88)
    GROUP_CASE(89)
    GROUP_CASE(90)
    GROUP_CASE(91)
    GROUP_CASE(92)
    GROUP_CASE(93)
    GROUP_CASE(94)
    GROUP_CASE(95)
    GROUP_CASE(96)
#endif
    default:
      printf("Adding group case to outer fluxm computation!\n");
      assert(false);
  }
#undef GROUP_CASE
}

__global__
void gpu_outer_convergence(const double *flux0_ptr, const double *flux0po_ptr,
                           ByteOffset flux0_offsets[3], 
                           ByteOffset flux0po_offsets[3], 
                           const double epsi, int *total_converged,
                           const int x_start, const int y_start, const int z_start)
{
  // We know there is never more than 32 warps in a CTA
  __shared__ int trampoline[32];

  const int x = blockIdx.x * blockDim.x + threadIdx.x + x_start;
  const int y = blockIdx.y * blockDim.y + threadIdx.y + y_start;
  const int z = blockIdx.z * blockDim.z + threadIdx.z + z_start;

  flux0_ptr += x * flux0_offsets[0] + y * flux0_offsets[1] + z * flux0_offsets[2];
  flux0po_ptr += x * flux0po_offsets[0] + y * flux0po_offsets[1] + z * flux0po_offsets[2];

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
  unsigned laneid;
  asm volatile("mov.u32 %0, %laneid;" : "=r"(laneid) : );
  unsigned warpid;
  asm volatile("mov.u32 %0, %warpid;" : "=r"(warpid) : );
  for (int i = 16; i >= 1; i/=2)
    local_converged += __shfl_xor(local_converged, i, 32);
  // Initialize the trampoline
  if (warpid == 0)
    trampoline[laneid] = 0;
  __syncthreads();
  // First thread in each warp writes out all values
  if (laneid == 0)
    trampoline[warpid] = local_converged;
  __syncthreads();
  // Butterfly reduction across all thread in the first warp
  if (warpid == 0) {
    local_converged = trampoline[laneid];
    for (int i = 16; i >= 1; i/=2)
      local_converged += __shfl_xor(local_converged, i, 32);
    // First thread does the atomic
    if (laneid == 0)
      atomicAdd(total_converged, local_converged);
  }
}

