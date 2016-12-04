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

__global__
void gpu_inner_source_single_moment(const MomentQuad  *sxs_ptr,
                                    const double      *flux0_ptr,
                                    const double      *q2grp0_ptr,
                                          MomentQuad  *qtot_ptr,
                                    ByteOffset        sxs_offsets[3],
                                    ByteOffset        flux0_offsets[3],
                                    ByteOffset        q2grp0_offsets[3],
                                    ByteOffset        qtot_offsets[3])
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;
  // Straight up data parallel so nothing interesting to do
  sxs_ptr += x * sxs_offsets[0] + y * sxs_offsets[1] + z * sxs_offsets[2];
  MomentQuad sxs_quad = *sxs_ptr;

  flux0_ptr += x * flux0_offsets[0] + y * flux0_offsets[1] + z * flux0_offsets[2];
  double flux0 = *flux0_ptr;

  q2grp0_ptr += x * q2grp0_offsets[0] + y * q2grp0_offsets[1] + z * q2grp0_offsets[2];
  double q0 = *q2grp0_ptr;

  MomentQuad quad;
  quad[0] = q0 + flux0 * sxs_quad[0]; 

  qtot_ptr += x * qtot_offsets[0] + y * qtot_offsets[1] + z * qtot_offsets[2];
  *qtot_ptr = quad;
}

__global__
void gpu_inner_source_multi_moment(const MomentQuad   *sxs_ptr,
                                   const double       *flux0_ptr,
                                   const double       *q2grp0_ptr,
                                   const MomentTriple *fluxm_ptr,
                                   const MomentTriple *q2grpm_ptr,
                                         MomentQuad   *qtot_ptr,
                                   ByteOffset         sxs_offsets[3],
                                   ByteOffset         flux0_offsets[3],
                                   ByteOffset         q2grp0_offsets[3],
                                   ByteOffset         fluxm_offsets[3],
                                   ByteOffset         q2grpm_offsets[3],
                                   ByteOffset         qtot_offsets[3],
                                   const int num_moments, const int lma[4])
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;
  // Straight up data parallel so nothing interesting to do
  sxs_ptr += x * sxs_offsets[0] + y * sxs_offsets[1] + z * sxs_offsets[2];
  MomentQuad sxs_quad = *sxs_ptr;

  flux0_ptr += x * flux0_offsets[0] + y * flux0_offsets[1] + z * flux0_offsets[2];
  double flux0 = *flux0_ptr;

  q2grp0_ptr += x * q2grp0_offsets[0] + y * q2grp0_offsets[1] + z * q2grp0_offsets[2];
  double q0 = *q2grp0_ptr;

  fluxm_ptr += x * fluxm_offsets[0] + y * fluxm_offsets[1] + z * fluxm_offsets[2];
  MomentTriple fluxm = *fluxm_ptr;

  q2grpm_ptr += x * q2grpm_offsets[0] + y * q2grpm_offsets[1] + z * q2grpm_offsets[2];
  MomentTriple qom = *q2grpm_ptr;

  MomentQuad quad;
  quad[0] = q0 + flux0 * sxs_quad[0]; 
  
  int moment = 0;
  for (int l = 1; l < num_moments; l++) {
    for (int i = 0; i < lma[l]; i++)
      quad[moment+i+1] = qom[moment+i] + fluxm[moment+i] * sxs_quad[l];
    moment += lma[l];
  }

  qtot_ptr += x * qtot_offsets[0] + y * qtot_offsets[1] + z * qtot_offsets[2];
  *qtot_ptr = quad;
}

__global__
void gpu_inner_convergence(const double *flux0_ptr, const double *flux0pi_ptr,
                           ByteOffset flux0_offsets[3],
                           ByteOffset flux0pi_offsets[3],
                           const double epsi, int *total_converged)
{
  // We know there is never more than 32 warps in a CTA
  __shared__ int trampoline[32];

  flux0_ptr += (blockIdx.x * blockDim.x + threadIdx.x) * flux0_offsets[0] + 
    (blockIdx.y * blockDim.y + threadIdx.y) * flux0_offsets[1] + 
    (blockIdx.z * blockDim.z + threadIdx.z) * flux0_offsets[2];
  flux0pi_ptr += (blockIdx.x * blockDim.x + threadIdx.x) * flux0pi_offsets[0] +
    (blockIdx.y * blockDim.y + threadIdx.y) * flux0pi_offsets[1] + 
    (blockIdx.z * blockDim.z + threadIdx.z) * flux0pi_offsets[2];

  const double tolr = 1.0e-12;

  double flux0pi = *flux0pi_ptr;
  double df = 1.0;
  if (fabs(flux0pi) < tolr) {
    flux0pi = 1.0;
    df = 0.0;
  }
  double flux0 = *flux0_ptr;
  df = fabs( (flux0 / flux0pi) - df );
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

