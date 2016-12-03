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

using namespace LegionRuntime::Accessor;

template<int GROUPS, int STRIP_SIZE>
__global__
void gpu_flux0_outer_source(const double      *qi0_ptrs[GROUPS], 
                            const double      *flux0_ptrs[GROUPS],
                            const MomentQuad  *slgg_ptrs[GROUPS],
                            const double      *mat_ptr,
                                  double      *qo0_ptrs[GROUPS],
                            ByteOffset        qi0_offsets[3],
                            ByteOffset        flux0_offsets[3],
                            ByteOffset        slgg_offsets[2],
                            ByteOffset        mat_offsets[3],
                            ByteOffset        qo0_offsets[3])
{
  __shared__ double flux_buffer[GROUPS][STRIP_SIZE];
  const int x = blockIdx.x * STRIP_SIZE + threadIdx.x;
  const int y = blockIdx.y;
  const int z = blockIdx.z;
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

template<int GROUPS, int STRIP_SIZE>
__global__
void gpu_fluxm_outer_source(const MomentTriple  *fluxm_ptrs[GROUPS],
                            const MomentQuad    *slgg_ptrs[GROUPS],
                            const double        *mat_ptr,
                                  MomentTriple  *qom_ptrs[GROUPS],
                            ByteOffset          fluxm_offsets[3],
                            ByteOffset          slgg_offsets[2],
                            ByteOffset          mat_offsets[3],
                            ByteOffset          qom_offsets[3],
                            const int           num_moments,
                            const int           lma[4])
{
  __shared__ MomentTriple fluxm_buffer[GROUPS][STRIP_SIZE];
  const int x = blockIdx.x * STRIP_SIZE + threadIdx.x;
  const int y = blockIdx.y;
  const int z = blockIdx.z;
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
  fluxm_buffer[group][x] = fluxm;
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
    fluxm = fluxm_buffer[g][x];
    for (int l = 0; l < (num_moments-1); l++)
      qom[l] += csm[l] * fluxm[l];
  }
  // Now we can write out the result
  asm volatile("st.global.cs.v2.f64 [%0], {%1,%2};" : : "l"(qom_ptr), 
                "d"(qom[0]), "d"(qom[1]) : "memory");
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(((char*)qom_ptr)+16),
                "d"(qom[2]) : "memory");
}

__global__
void gpu_outer_convergence(const double *flux0_ptr, const double *flux0po_ptr,
                           ByteOffset flux0_offsets[3], 
                           ByteOffset flux0po_offsets[3], 
                           const double epsi, int *total_converged)
{
  // We know there is never more than 32 warps in a CTA
  __shared__ int trampoline[32];

  flux0_ptr += (blockIdx.x * blockDim.x + threadIdx.x) * flux0_offsets[0] + 
    (blockIdx.y * blockDim.y + threadIdx.y) * flux0_offsets[1] + 
    (blockIdx.z * blockDim.z + threadIdx.z) * flux0_offsets[2];
  flux0po_ptr += (blockIdx.x * blockDim.x + threadIdx.x) * flux0po_offsets[0] +
    (blockIdx.y * blockDim.y + threadIdx.y) * flux0po_offsets[1] + 
    (blockIdx.z * blockDim.z + threadIdx.z) * flux0po_offsets[2];

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

