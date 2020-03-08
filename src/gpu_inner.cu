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

__global__
void gpu_inner_source_single_moment(const Point<3> origin,
                                    const AccessorRO<MomentQuad,3> fa_sxs,
                                    const AccessorRO<double,3> fa_flux0,
                                    const AccessorRO<double,3> fa_q2grp0,
                                    const AccessorWO<MomentQuad,3> fa_qtot)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;
  const Point<3> p = origin + Point<3>(x,y,z);

  // Straight up data parallel so nothing interesting to do
  MomentQuad sxs_quad = fa_sxs[p];
  double flux0 = fa_flux0[p];
  double q0 = fa_q2grp0[p];

  MomentQuad quad;
  quad[0] = q0 + flux0 * sxs_quad[0]; 
  fa_qtot[p] = quad;
}

__host__
void run_inner_source_single_moment(const Rect<3> subgrid_bounds,
                                    const AccessorRO<MomentQuad,3> fa_sxs,
                                    const AccessorRO<double,3> fa_flux0,
                                    const AccessorRO<double,3> fa_q2grp0,
                                    const AccessorWO<MomentQuad,3> fa_qtot)
{
  const int x_range = (subgrid_bounds.hi[0] - subgrid_bounds.lo[0]) + 1;
  const int y_range = (subgrid_bounds.hi[1] - subgrid_bounds.lo[1]) + 1;
  const int z_range = (subgrid_bounds.hi[2] - subgrid_bounds.lo[2]) + 1;

  dim3 block(gcd(x_range,32),gcd(y_range,4),gcd(z_range,4));
  dim3 grid(x_range/block.x, y_range/block.y, z_range/block.z);
  gpu_inner_source_single_moment<<<grid,block>>>(subgrid_bounds.lo,
                                                 fa_sxs, fa_flux0, 
                                                 fa_q2grp0, fa_qtot);
}

__global__
void gpu_inner_source_multi_moment(const Point<3> origin,
                                   const AccessorRO<MomentQuad,3> fa_sxs,
                                   const AccessorRO<double,3> fa_flux0,
                                   const AccessorRO<double,3> fa_q2grp0,
                                   const AccessorRO<MomentTriple,3> fa_fluxm,
                                   const AccessorRO<MomentTriple,3> fa_q2grpm,
                                   const AccessorWO<MomentQuad,3> fa_qtot,
                                   const int num_moments,
                                   const ConstBuffer<4,int> lma)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;
  const Point<3> p = origin + Point<3>(x,y,z);
  // Straight up data parallel so nothing interesting to do 
  MomentQuad sxs_quad = fa_sxs[p];
  double flux0 = fa_flux0[p];
  double q0 = fa_q2grp0[p];
  MomentTriple fluxm = fa_fluxm[p];
  MomentTriple qom = fa_q2grpm[p];

  MomentQuad quad;
  quad[0] = q0 + flux0 * sxs_quad[0]; 
  
  int moment = 0;
  for (int l = 1; l < num_moments; l++) {
    for (int i = 0; i < lma[l]; i++)
      quad[moment+i+1] = qom[moment+i] + fluxm[moment+i] * sxs_quad[l];
    moment += lma[l];
  }

  fa_qtot[p] = quad;
}

__host__
void run_inner_source_multi_moment(const Rect<3> subgrid_bounds,
                                   const AccessorRO<MomentQuad,3> fa_sxs,
                                   const AccessorRO<double,3> fa_flux0,
                                   const AccessorRO<double,3> fa_q2grp0,
                                   const AccessorRO<MomentTriple,3> fa_fluxm,
                                   const AccessorRO<MomentTriple,3> fa_q2grpm,
                                   const AccessorWO<MomentQuad,3> fa_qtot,
                                   const int num_moments, const int lma[4])
{
  const int x_range = (subgrid_bounds.hi[0] - subgrid_bounds.lo[0]) + 1;
  const int y_range = (subgrid_bounds.hi[1] - subgrid_bounds.lo[1]) + 1;
  const int z_range = (subgrid_bounds.hi[2] - subgrid_bounds.lo[2]) + 1;

  dim3 block(gcd(x_range,32),gcd(y_range,4),gcd(z_range,4));
  dim3 grid(x_range/block.x, y_range/block.y, z_range/block.z);
  gpu_inner_source_multi_moment<<<grid,block>>>(subgrid_bounds.lo,
                                                fa_sxs, fa_flux0, fa_q2grp0,
                                                fa_fluxm, fa_q2grpm, fa_qtot,
                                                num_moments, ConstBuffer<4,int>(lma));
}

__global__
void gpu_inner_convergence(const Point<3> origin,
                           const AccessorRO<double,3> fa_flux0,
                           const AccessorRO<double,3> fa_flux0pi,
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

  const double tolr = 1.0e-12;

  double flux0pi = fa_flux0pi[p];
  double df = 1.0;
  if (fabs(flux0pi) < tolr) {
    flux0pi = 1.0;
    df = 0.0;
  }
  double flux0 = fa_flux0[p];
  df = fabs( (flux0 / flux0pi) - df );
  int local_converged = 1;
  if (df > epsi)
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
void gpu_sum_inner_convergence(const DeferredBuffer<int,1> buffer,
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
void run_inner_convergence(const Rect<3> subgrid_bounds,
                           const DeferredValue<bool> &result,
                           const std::vector<AccessorRO<double,3> > &fa_flux0,
                           const std::vector<AccessorRO<double,3> > &fa_flux0pi,
                           const double epsi)
{
  // Launch the kernels
  const int x_range = (subgrid_bounds.hi[0] - subgrid_bounds.lo[0]) + 1;
  const int y_range = (subgrid_bounds.hi[1] - subgrid_bounds.lo[1]) + 1;
  const int z_range = (subgrid_bounds.hi[2] - subgrid_bounds.lo[2]) + 1;

  dim3 block(gcd(x_range,32),gcd(y_range,4),gcd(z_range,4));
  dim3 grid(x_range/block.x, y_range/block.y, z_range/block.z);
  const size_t total_blocks = grid.x*grid.y*grid.z;
  assert(fa_flux0.size() == fa_flux0pi.size());
  const Rect<1> bounds(Point<1>(0),Point<1>(total_blocks * fa_flux0.size() - 1));
  DeferredBuffer<int,1> buffer(bounds, Memory::GPU_FB_MEM);
  for (unsigned idx = 0; idx < fa_flux0.size(); idx++) {
    gpu_inner_convergence<<<grid,block>>>(subgrid_bounds.lo,
                                          fa_flux0[idx], fa_flux0pi[idx],
                                          epsi, buffer, idx * total_blocks);
  }
  dim3 block2((bounds.hi[0]+1) > 1024 ? 1024 : (bounds.hi[0]+1),1,1);
  // Round up to the nearest multiple of warps
  while ((block2.x % 32) != 0)
    block2.x++;
  dim3 grid2(1,1,1);
  const int expected = x_range * y_range * z_range * fa_flux0.size();
  gpu_sum_inner_convergence<<<grid2,block2>>>(buffer, result, bounds.hi[0]+1, expected);
}

