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

// Assume no more than 2K angles
__device__ double device_ec[2048* 4 * 8];
__device__ double device_mu[2048];
__device__ double device_eta[2048];
__device__ double device_xi[2048];
__device__ double device_w[2048];

__device__ __forceinline__
ByteOffset operator*(const ByteOffset offsets[3], const Point<3> &point)
{
  return (offsets[0] * point.x[0] + offsets[1] * point.x[1] + offsets[2] * point.x[2]);
}

__device__ __forceinline__
void atomicAdd(double *ptr, double value)
{
  unsigned long long int* address_as_ull = (unsigned long long int*)ptr; 
  unsigned long long int old = *address_as_ull, assumed; 
  do { 
    assumed = old; 
    old = atomicCAS(address_as_ull, assumed, 
        __double_as_longlong(value + __longlong_as_double(assumed))); 
  } while (assumed != old);
}

__device__ __forceinline__
double* get_angle_ptr(void *ptr, const ByteOffset offset[3],
                      const Point<3> &point, int num_angles, int ang)
{
  double *result = (double*)ptr;
  result += ((offset * point) * num_angles);
  result += ang * blockDim.x + threadIdx.x;
  return result;
}

template<int THR_ANGLES>
__global__
void gpu_time_dependent_sweep_with_fixup(const Point<3> origin, 
                                         const MomentQuad *qtot_ptr,
                                               double     *flux_ptr,
                                               MomentQuad *fluxm_ptr,
                                         const double     *dinv_ptr,
                                         const double     *time_flux_in_ptr,
                                               double     *time_flux_out_ptr,
                                         const double     *t_xs_ptr,
                                               double     *ghostx_out_ptr,
                                               double     *ghosty_out_ptr,
                                               double     *ghostz_out_ptr,
                                         const double     *qim_ptr,
                                         const double     *ghostx_in_ptr,
                                         const double     *ghosty_in_ptr,
                                         const double     *ghostz_in_ptr,
                                         const ByteOffset qtot_offsets[3],
                                         const ByteOffset flux_offsets[3],
                                         const ByteOffset fluxm_offsets[3],
                                         const ByteOffset dinv_offsets[3],
                                         const ByteOffset time_flux_in_offsets[3],
                                         const ByteOffset time_flux_out_offsets[3],
                                         const ByteOffset t_xs_offsets[3],
                                         const ByteOffset ghostx_out_offsets[3],
                                         const ByteOffset ghosty_out_offsets[3],
                                         const ByteOffset ghostz_out_offsets[3],
                                         const ByteOffset qim_offsets[3],
                                         const ByteOffset ghostx_in_offsets[3],
                                         const ByteOffset ghosty_in_offsets[3],
                                         const ByteOffset ghostz_in_offsets[3],
                                         const int x_range, const int y_range, 
                                         const int z_range, const int corner,
                                         const bool stride_x_positive,
                                         const bool stride_y_positive,
                                         const bool stride_z_positive,
                                         const bool mms_source, 
                                         const int num_moments, 
                                         const double hi, const double hj,
                                         const double hk, const double vdelt,
                                         double *yflux_pencil,
                                         double *zflux_plane)
{
  __shared__ int int_trampoline[32];
  __shared__ double double_trampoline[32];

  double psi[THR_ANGLES];
  double pc[THR_ANGLES];
  double psii[THR_ANGLES];
  double psij[THR_ANGLES];
  double psik[THR_ANGLES];
  double hv_x[THR_ANGLES];
  double hv_y[THR_ANGLES];
  double hv_z[THR_ANGLES];
  double hv_t[THR_ANGLES];
  double fx_hv_x[THR_ANGLES];
  double fx_hv_y[THR_ANGLES];
  double fx_hv_z[THR_ANGLES];
  double fx_hv_t[THR_ANGLES];
  double time_flux_in[THR_ANGLES];

  const int num_angles = THR_ANGLES * blockDim.x;
  const int corner_offset = corner * num_angles * num_moments; 

  unsigned laneid;
  asm volatile("mov.u32 %0, %laneid;" : "=r"(laneid) : );
  unsigned warpid;
  asm volatile("mov.u32 %0, %warpid;" : "=r"(warpid) : );

  const double tolr = 1.0e-12;

  for (int z = 0; z < z_range; z++) {
    for (int y = 0; y < y_range; y++) {
      for (int x = 0; x < x_range; x++) {
        // Figure out the local point that we are working on    
        Point<3> local_point = origin;
        if (stride_x_positive)
          local_point.x[0] += x;
        else
          local_point.x[0] -= x;
        if (stride_y_positive)
          local_point.x[1] += y;
        else
          local_point.x[1] -= y;
        if (stride_z_positive)
          local_point.x[2] += z;
        else
          local_point.x[2] -= z;

        // Compute the angular source
        MomentQuad quad = *(qtot_ptr + qtot_offsets * local_point);
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          psi[ang] = quad[0];
        if (num_moments > 1) {
          for (int l = 1; l < num_moments; l++) {
            const int moment_offset = corner_offset + l * num_angles;
            #pragma unroll
            for (int ang = 0; ang < THR_ANGLES; ang++) {
              psi[ang] += device_ec[moment_offset+ang*blockDim.x+threadIdx.x] * quad[l];
            }
          }
        }

        // If we're doing MMS
        if (mms_source) {
          #pragma unroll
          for (int ang = 0; ang < THR_ANGLES; ang++)
            psi[ang] += *(qim_ptr + ang*blockDim.x + threadIdx.x);
        }

        // Compute the initial solution
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          pc[ang] = psi[ang];
        // X ghost cells
        if (stride_x_positive) {
          // reading from x-1 
          Point<3> ghost_point = local_point;
          ghost_point.x[0] -= 1;
          if (x == 0) {
            // Ghost cell array
            #pragma unroll 
            for (int ang = 0; ang < THR_ANGLES; ang++)
              psii[ang] = *((ghostx_in_ptr + ghostx_in_offsets * ghost_point) + 
                            ang*blockDim.x + threadIdx.x);
          } 
          // Else nothing: psii already contains next flux
        } else {
          // reading from x+1
          Point<3> ghost_point = local_point;
          ghost_point.x[0] += 1;
          // Local coordinates here
          if (x == 0) {
            // Ghost cell array
            #pragma unroll
            for (int ang = 0; ang < THR_ANGLES; ang++)
              psii[ang] = *((ghostx_in_ptr + ghostx_in_offsets * ghost_point) +
                            ang*blockDim.x + threadIdx.x);
          }
          // Else nothing: psii already contains next flux
        }
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          pc[ang] += psii[ang] * device_mu[ang*blockDim.x + threadIdx.x] * hi;
        // Y ghost cells
        if (stride_y_positive) {
          // reading from y-1
          Point<3> ghost_point = local_point;
          ghost_point.x[1] -= 1;
          if (y == 0) {
            // Ghost cell array
            #pragma unroll
            for (int ang = 0; ang < THR_ANGLES; ang++)
              psij[ang] = *((ghosty_in_ptr + ghosty_in_offsets * ghost_point) +
                            ang*blockDim.x + threadIdx.x);
          } else {
            // Local array
            #pragma unroll
            for (int ang = 0; ang < THR_ANGLES; ang++)
              psij[ang] = yflux_pencil[x * num_angles + ang * blockDim.x + threadIdx.x];
          }
        } else {
          // reading from y+1
          Point<3> ghost_point = local_point;
          ghost_point.x[1] += 1;
          // Local coordinates here
          if (y == 0) {
            // Ghost cell array
            #pragma unroll
            for (int ang = 0; ang < THR_ANGLES; ang++)
              psij[ang] = *((ghosty_in_ptr + ghosty_in_offsets * ghost_point) +
                            ang*blockDim.x + threadIdx.x);
          } else {
            // Local array
            #pragma unroll
            for (int ang = 0; ang < THR_ANGLES; ang++)
              psij[ang] = yflux_pencil[x * num_angles + ang * blockDim.x + threadIdx.x];
          }
        }
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          pc[ang] += psij[ang] * device_eta[ang * blockDim.x + threadIdx.x] * hj;
        // Z ghost cells
        if (stride_z_positive) {
          // reading from z-1
          Point<3> ghost_point = local_point;
          ghost_point.x[2] -= 1;
          if (z == 0) {
            // Ghost cell array
            #pragma unroll
            for (int ang = 0; ang < THR_ANGLES; ang++)
              psik[ang] = *((ghostz_in_ptr + ghostz_in_offsets * ghost_point) +
                            ang*blockDim.x + threadIdx.x);
          } else {
            // Local array
            const int offset = (y * x_range + x) * num_angles;
            #pragma unroll
            for (int ang = 0; ang < THR_ANGLES; ang++)
              psik[ang] = zflux_plane[offset + ang * blockDim.x + threadIdx.x];
          }
        } else {
          // reading from z+1
          Point<3> ghost_point = local_point;
          ghost_point.x[2] += 1;
          // Local coordinates here
          if (z == 0) {
            // Ghost cell array
            #pragma unroll
            for (int ang = 0; ang < THR_ANGLES; ang++)
              psik[ang] = *((ghostz_in_ptr + ghostz_in_offsets * ghost_point) + 
                            ang*blockDim.x + threadIdx.x);
          } else {
            // Local array
            const int offset = (y * x_range + x) * num_angles;
            #pragma unroll
            for (int ang = 0; ang < THR_ANGLES; ang++)
              psik[ang] = zflux_plane[offset + ang * blockDim.x + threadIdx.x];
          }
        }
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          pc[ang] += psik[ang] * device_xi[ang * blockDim.x + threadIdx.x] * hk;

        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++) {
          time_flux_in[ang] = *((time_flux_in_ptr + time_flux_in_offsets * local_point) +
                                ang * blockDim.x + threadIdx.x);
          pc[ang] += vdelt * time_flux_in[ang];
        }
        
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++) {
          double dinv = *((dinv_ptr + dinv_offsets * local_point) + 
                          ang * blockDim.x + threadIdx.x);
          pc[ang] *= dinv;
        }

        // DO THE FIXUP
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          hv_x[ang] = 1.0;
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          hv_y[ang] = 1.0;
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          hv_z[ang] = 1.0;
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          hv_t[ang] = 1.0;

        const double t_xs = *(t_xs_ptr + t_xs_offsets * local_point);
        int old_negative_fluxes = 0;
        while (true) {
          unsigned negative_fluxes = 0;
          // Figure out how many negative fluxes we have
          #pragma unroll
          for (int ang = 0; ang < THR_ANGLES; ang++) {
            fx_hv_x[ang] = 2.0 * pc[ang] - psii[ang];
            if (fx_hv_x[ang] < 0.0) {
              hv_x[ang] = 0.0;
              negative_fluxes++;
            }
          }
          #pragma unroll
          for (int ang = 0; ang < THR_ANGLES; ang++) {
            fx_hv_y[ang] = 2.0 * pc[ang] - psij[ang];
            if (fx_hv_y[ang] < 0.0) {
              hv_y[ang] = 0.0;
              negative_fluxes++;
            }
          }
          #pragma unroll
          for (int ang = 0; ang < THR_ANGLES; ang++) {
            fx_hv_z[ang] = 2.0 * pc[ang] - psik[ang];
            if (fx_hv_z[ang] < 0.0) {
              hv_z[ang] = 0.0;
              negative_fluxes++;
            }
          }
          #pragma unroll
          for (int ang = 0; ang < THR_ANGLES; ang++) {
            fx_hv_t[ang] = 2.0 * pc[ang] - time_flux_in[ang];
            if (fx_hv_t[ang] < 0.0) {
              hv_t[ang] = 0.0;
              negative_fluxes++;
            }
          }
          // CTA-wide reduction
          #pragma unroll
          for (int i = 16; i >= 1; i /= 2)
            negative_fluxes += __shfl_xor(negative_fluxes, i, 32);
          // Initialize
          if (warpid == 0)
            int_trampoline[laneid] = 0;
          __syncthreads();
          if (laneid == 0)
            int_trampoline[warpid] = negative_fluxes;
          __syncthreads();
          negative_fluxes = int_trampoline[laneid];
          #pragma unroll
          for (int i = 16; i >= 1; i /= 2)
            negative_fluxes += __shfl_xor(negative_fluxes, i, 32);
          // All threads have the same negative flux count now
          if (negative_fluxes == old_negative_fluxes)
            break;
          old_negative_fluxes = negative_fluxes;
          #pragma unroll
          for (int ang = 0; ang < THR_ANGLES; ang++) {
            pc[ang] = psi[ang] + 0.5 * (
              psii[ang] * device_mu[ang*blockDim.x + threadIdx.x] * hi * (1.0 + hv_x[ang]) +
              psij[ang] * device_eta[ang*blockDim.x + threadIdx.x] * hj * (1.0 + hv_y[ang]) +
              psik[ang] * device_xi[ang*blockDim.x + threadIdx.x] * hk * (1.0 + hv_z[ang]) +
              time_flux_in[ang] * vdelt * (1.0 + hv_t[ang]) );
            double den = (pc[ang] <= 0.0) ? 0.0 : (t_xs + 
                device_mu[ang*blockDim.x + threadIdx.x] * hi * hv_x[ang] + 
                device_eta[ang*blockDim.x + threadIdx.x] * hj * hv_y[ang] + 
                device_xi[ang*blockDim.x + threadIdx.x] * hk * hv_z[ang] +
                vdelt * hv_t[ang]);
            if (den < tolr)
              pc[ang] = 0.0;
            else
              pc[ang] /= den;
          }
        }
        // Fixup done so compute the update values
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          psii[ang] = fx_hv_x[ang] * hv_x[ang];
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          psij[ang] = fx_hv_y[ang] * hv_y[ang];
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          psik[ang] = fx_hv_z[ang] * hv_z[ang];
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++) {
          double time_flux_out = fx_hv_t[ang] * hv_t[ang];
          *((time_flux_out_ptr + time_flux_out_offsets * local_point) +
              ang * blockDim.x + threadIdx.x) = time_flux_out;
        }
        // Write out the ghost regions
        // X ghost
        if (x == (x_range - 1)) {
          #pragma unroll
          for (int ang = 0; ang < THR_ANGLES; ang++)
            *((ghostx_out_ptr + ghostx_out_offsets * local_point) +
                ang * blockDim.x + threadIdx.x) = psii[ang];
        }
        // Y ghost
        if (y == (y_range - 1)) {
          #pragma unroll
          for (int ang = 0; ang < THR_ANGLES; ang++)
            *((ghosty_out_ptr + ghosty_out_offsets * local_point) + 
                ang * blockDim.x + threadIdx.x) = psij[ang];
        } else {
          // Write to the pencil
          const int offset = x * num_angles;
          #pragma unroll
          for (int ang = 0; ang < THR_ANGLES; ang++)
            yflux_pencil[offset + ang * blockDim.x + threadIdx.x] = psij[ang]; 
        }
        // Z ghost
        if (z == (z_range - 1)) {
          #pragma unroll
          for (int ang = 0; ang < THR_ANGLES; ang++)
            *((ghostz_out_ptr + ghostz_out_offsets * local_point) + 
                ang * blockDim.x + threadIdx.x) = psik[ang];
        } else {
          const int offset = (y * x_range + x) * num_angles;
          #pragma unroll
          for (int ang = 0; ang < THR_ANGLES; ang++)
            zflux_plane[offset + ang * blockDim.x + threadIdx.x] = psik[ang];
        }
        // Finally we apply reductions to the flux moments
        double total = 0.0;  
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++) {
          psi[ang] = device_w[ang * blockDim.x + threadIdx.x] * pc[ang];
          total += psi[ang];
        }
        // CTA-wide reduction to one warp and then down to one thread
        #pragma unroll
        for (int i = 16; i >= 1; i /= 2) {
          int hi_part = __shfl_xor(__double2hiint(total), i, 32);
          int lo_part = __shfl_xor(__double2loint(total), i, 32);
          total += __hiloint2double(hi_part,lo_part); 
        }
        if (warpid == 0)
          double_trampoline[laneid] = 0.0;
        __syncthreads();
        if (laneid == 0)
          double_trampoline[warpid] = total;
        __syncthreads();
        if (warpid == 0) {
          total = double_trampoline[laneid];
          #pragma unroll
          for (int i = 16; i >= 1; i /= 2) {
            int hi_part = __shfl_xor(__double2hiint(total), i, 32);
            int lo_part = __shfl_xor(__double2loint(total), i, 32);
            total += __hiloint2double(hi_part,lo_part); 
          }
          // Do the reduction
          if (laneid == 0) {
            double *local_flux = flux_ptr + flux_offsets * local_point;
            atomicAdd(local_flux, total);
          }
        }
        if (num_moments > 1) {
          for (int l = 1; l < num_moments; l++) {
            const int offset = l * num_angles + corner * num_angles * num_moments;
            total = 0.0;
            #pragma unroll
            for (int ang = 0; ang < THR_ANGLES; ang++)
              total += device_ec[offset + ang] * psi[ang];
            __syncthreads();
            if (warpid == 0)
              double_trampoline[laneid] = 0.0;
            __syncthreads();
            if (laneid == 0)
              double_trampoline[warpid] = total;
            __syncthreads();
            if (warpid == 0)
              total = double_trampoline[laneid];
            #pragma unroll
            for (int i = 16; i >= 1; i /= 2) {
              int hi_part = __shfl_xor(__double2hiint(total), i, 32);
              int lo_part = __shfl_xor(__double2loint(total), i, 32);
              total += __hiloint2double(hi_part,lo_part); 
            }
            if (laneid == 0) {
              double *local_fluxm = (double*)(fluxm_ptr + fluxm_offsets * local_point);
              local_fluxm += (l-1);
              atomicAdd(local_fluxm, total);
            }
          }
        }
      }
    }
  }
}

template<int THR_ANGLES>
__global__
void gpu_time_dependent_sweep_without_fixup(const Point<3> origin, 
                                         const MomentQuad *qtot_ptr,
                                               double     *flux_ptr,
                                               MomentQuad *fluxm_ptr,
                                         const double     *dinv_ptr,
                                         const double     *time_flux_in_ptr,
                                               double     *time_flux_out_ptr,
                                               double     *ghostx_out_ptr,
                                               double     *ghosty_out_ptr,
                                               double     *ghostz_out_ptr,
                                         const double     *qim_ptr,
                                         const double     *ghostx_in_ptr,
                                         const double     *ghosty_in_ptr,
                                         const double     *ghostz_in_ptr,
                                         const ByteOffset qtot_offsets[3],
                                         const ByteOffset flux_offsets[3],
                                         const ByteOffset fluxm_offsets[3],
                                         const ByteOffset dinv_offsets[3],
                                         const ByteOffset time_flux_in_offsets[3],
                                         const ByteOffset time_flux_out_offsets[3],
                                         const ByteOffset ghostx_out_offsets[3],
                                         const ByteOffset ghosty_out_offsets[3],
                                         const ByteOffset ghostz_out_offsets[3],
                                         const ByteOffset qim_offsets[3],
                                         const ByteOffset ghostx_in_offsets[3],
                                         const ByteOffset ghosty_in_offsets[3],
                                         const ByteOffset ghostz_in_offsets[3],
                                         const int x_range, const int y_range, 
                                         const int z_range, const int corner,
                                         const bool stride_x_positive,
                                         const bool stride_y_positive,
                                         const bool stride_z_positive,
                                         const bool mms_source, 
                                         const int num_moments, 
                                         const double hi, const double hj,
                                         const double hk, const double vdelt,
                                         double *yflux_pencil,
                                         double *zflux_plane)
{
  __shared__ double double_trampoline[32];

  double psi[THR_ANGLES];
  double pc[THR_ANGLES];
  double psii[THR_ANGLES];
  double psij[THR_ANGLES];
  double psik[THR_ANGLES];
  double time_flux_in[THR_ANGLES];

  const int num_angles = THR_ANGLES * blockDim.x;
  const int corner_offset = corner * num_angles * num_moments; 

  unsigned laneid;
  asm volatile("mov.u32 %0, %laneid;" : "=r"(laneid) : );
  unsigned warpid;
  asm volatile("mov.u32 %0, %warpid;" : "=r"(warpid) : );

  for (int z = 0; z < z_range; z++) {
    for (int y = 0; y < y_range; y++) {
      for (int x = 0; x < x_range; x++) {
        // Figure out the local point that we are working on    
        Point<3> local_point = origin;
        if (stride_x_positive)
          local_point.x[0] += x;
        else
          local_point.x[0] -= x;
        if (stride_y_positive)
          local_point.x[1] += y;
        else
          local_point.x[1] -= y;
        if (stride_z_positive)
          local_point.x[2] += z;
        else
          local_point.x[2] -= z;

        // Compute the angular source
        MomentQuad quad = *(qtot_ptr + qtot_offsets * local_point);
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          psi[ang] = quad[0];
        if (num_moments > 1) {
          for (int l = 1; l < num_moments; l++) {
            const int moment_offset = corner_offset + l * num_angles;
            #pragma unroll
            for (int ang = 0; ang < THR_ANGLES; ang++) {
              psi[ang] += device_ec[moment_offset+ang*blockDim.x+threadIdx.x] * quad[l];
            }
          }
        }

        // If we're doing MMS
        if (mms_source) {
          #pragma unroll
          for (int ang = 0; ang < THR_ANGLES; ang++)
            psi[ang] += *(qim_ptr + ang*blockDim.x + threadIdx.x);
        }

        // Compute the initial solution
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          pc[ang] = psi[ang];
        // X ghost cells
        if (stride_x_positive) {
          // reading from x-1 
          Point<3> ghost_point = local_point;
          ghost_point.x[0] -= 1;
          if (x == 0) {
            // Ghost cell array
            #pragma unroll 
            for (int ang = 0; ang < THR_ANGLES; ang++)
              psii[ang] = *((ghostx_in_ptr + ghostx_in_offsets * ghost_point) + 
                            ang*blockDim.x + threadIdx.x);
          } 
          // Else nothing: psii already contains next flux
        } else {
          // reading from x+1
          Point<3> ghost_point = local_point;
          ghost_point.x[0] += 1;
          // Local coordinates here
          if (x == 0) {
            // Ghost cell array
            #pragma unroll
            for (int ang = 0; ang < THR_ANGLES; ang++)
              psii[ang] = *((ghostx_in_ptr + ghostx_in_offsets * ghost_point) +
                            ang*blockDim.x + threadIdx.x);
          }
          // Else nothing: psii already contains next flux
        }
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          pc[ang] += psii[ang] * device_mu[ang*blockDim.x + threadIdx.x] * hi;
        // Y ghost cells
        if (stride_y_positive) {
          // reading from y-1
          Point<3> ghost_point = local_point;
          ghost_point.x[1] -= 1;
          if (y == 0) {
            // Ghost cell array
            #pragma unroll
            for (int ang = 0; ang < THR_ANGLES; ang++)
              psij[ang] = *((ghosty_in_ptr + ghosty_in_offsets * ghost_point) +
                            ang*blockDim.x + threadIdx.x);
          } else {
            // Local array
            #pragma unroll
            for (int ang = 0; ang < THR_ANGLES; ang++)
              psij[ang] = yflux_pencil[x * num_angles + ang * blockDim.x + threadIdx.x];
          }
        } else {
          // reading from y+1
          Point<3> ghost_point = local_point;
          ghost_point.x[1] += 1;
          // Local coordinates here
          if (y == 0) {
            // Ghost cell array
            #pragma unroll
            for (int ang = 0; ang < THR_ANGLES; ang++)
              psij[ang] = *((ghosty_in_ptr + ghosty_in_offsets * ghost_point) +
                            ang*blockDim.x + threadIdx.x);
          } else {
            // Local array
            #pragma unroll
            for (int ang = 0; ang < THR_ANGLES; ang++)
              psij[ang] = yflux_pencil[x * num_angles + ang * blockDim.x + threadIdx.x];
          }
        }
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          pc[ang] += psij[ang] * device_eta[ang * blockDim.x + threadIdx.x] * hj;
        // Z ghost cells
        if (stride_z_positive) {
          // reading from z-1
          Point<3> ghost_point = local_point;
          ghost_point.x[2] -= 1;
          if (z == 0) {
            // Ghost cell array
            #pragma unroll
            for (int ang = 0; ang < THR_ANGLES; ang++)
              psik[ang] = *((ghostz_in_ptr + ghostz_in_offsets * ghost_point) +
                            ang*blockDim.x + threadIdx.x);
          } else {
            // Local array
            const int offset = (y * x_range + x) * num_angles;
            #pragma unroll
            for (int ang = 0; ang < THR_ANGLES; ang++)
              psik[ang] = zflux_plane[offset + ang * blockDim.x + threadIdx.x];
          }
        } else {
          // reading from z+1
          Point<3> ghost_point = local_point;
          ghost_point.x[2] += 1;
          // Local coordinates here
          if (z == 0) {
            // Ghost cell array
            #pragma unroll
            for (int ang = 0; ang < THR_ANGLES; ang++)
              psik[ang] = *((ghostz_in_ptr + ghostz_in_offsets * ghost_point) + 
                            ang*blockDim.x + threadIdx.x);
          } else {
            // Local array
            const int offset = (y * x_range + x) * num_angles;
            #pragma unroll
            for (int ang = 0; ang < THR_ANGLES; ang++)
              psik[ang] = zflux_plane[offset + ang * blockDim.x + threadIdx.x];
          }
        }
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          pc[ang] += psik[ang] * device_xi[ang * blockDim.x + threadIdx.x] * hk;

        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++) {
          time_flux_in[ang] = *((time_flux_in_ptr + time_flux_in_offsets * local_point) +
                                ang * blockDim.x + threadIdx.x);
          pc[ang] += vdelt * time_flux_in[ang];
        }
        
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++) {
          double dinv = *((dinv_ptr + dinv_offsets * local_point) + 
                          ang * blockDim.x + threadIdx.x);
          pc[ang] *= dinv;
        }

        // NO FIXUP
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          psii[ang] = 2.0 * pc[ang] - psii[ang];
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          psij[ang] = 2.0 * pc[ang] - psij[ang];
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          psik[ang] = 2.0 * pc[ang] - psik[ang];

        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++) {
          double time_flux_out = 2.0 * pc[ang] - time_flux_in[ang];
          *((time_flux_out_ptr + time_flux_out_offsets * local_point) +
              ang * blockDim.x + threadIdx.x) = time_flux_out;
        }
        // Write out the ghost regions
        // X ghost
        if (x == (x_range - 1)) {
          #pragma unroll
          for (int ang = 0; ang < THR_ANGLES; ang++)
            *((ghostx_out_ptr + ghostx_out_offsets * local_point) +
                ang * blockDim.x + threadIdx.x) = psii[ang];
        }
        // Y ghost
        if (y == (y_range - 1)) {
          #pragma unroll
          for (int ang = 0; ang < THR_ANGLES; ang++)
            *((ghosty_out_ptr + ghosty_out_offsets * local_point) + 
                ang * blockDim.x + threadIdx.x) = psij[ang];
        } else {
          // Write to the pencil
          const int offset = x * num_angles;
          #pragma unroll
          for (int ang = 0; ang < THR_ANGLES; ang++)
            yflux_pencil[offset + ang * blockDim.x + threadIdx.x] = psij[ang]; 
        }
        // Z ghost
        if (z == (z_range - 1)) {
          #pragma unroll
          for (int ang = 0; ang < THR_ANGLES; ang++)
            *((ghostz_out_ptr + ghostz_out_offsets * local_point) + 
                ang * blockDim.x + threadIdx.x) = psik[ang];
        } else {
          const int offset = (y * x_range + x) * num_angles;
          #pragma unroll
          for (int ang = 0; ang < THR_ANGLES; ang++)
            zflux_plane[offset + ang * blockDim.x + threadIdx.x] = psik[ang];
        }
        // Finally we apply reductions to the flux moments
        double total = 0.0;  
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++) {
          psi[ang] = device_w[ang * blockDim.x + threadIdx.x] * pc[ang];
          total += psi[ang];
        }
        // CTA-wide reduction to one warp and then down to one thread
        #pragma unroll
        for (int i = 16; i >= 1; i /= 2) {
          int hi_part = __shfl_xor(__double2hiint(total), i, 32);
          int lo_part = __shfl_xor(__double2loint(total), i, 32);
          total += __hiloint2double(hi_part,lo_part); 
        }
        if (warpid == 0)
          double_trampoline[laneid] = 0.0;
        __syncthreads();
        if (laneid == 0)
          double_trampoline[warpid] = total;
        __syncthreads();
        if (warpid == 0) {
          total = double_trampoline[laneid];
          #pragma unroll
          for (int i = 16; i >= 1; i /= 2) {
            int hi_part = __shfl_xor(__double2hiint(total), i, 32);
            int lo_part = __shfl_xor(__double2loint(total), i, 32);
            total += __hiloint2double(hi_part,lo_part); 
          }
          // Do the reduction
          if (laneid == 0) {
            double *local_flux = flux_ptr + flux_offsets * local_point;
            atomicAdd(local_flux, total);
          }
        }
        if (num_moments > 1) {
          for (int l = 1; l < num_moments; l++) {
            const int offset = l * num_angles + corner * num_angles * num_moments;
            total = 0.0;
            #pragma unroll
            for (int ang = 0; ang < THR_ANGLES; ang++)
              total += device_ec[offset + ang] * psi[ang];
            __syncthreads();
            if (warpid == 0)
              double_trampoline[laneid] = 0.0;
            __syncthreads();
            if (laneid == 0)
              double_trampoline[warpid] = total;
            __syncthreads();
            if (warpid == 0)
              total = double_trampoline[laneid];
            #pragma unroll
            for (int i = 16; i >= 1; i /= 2) {
              int hi_part = __shfl_xor(__double2hiint(total), i, 32);
              int lo_part = __shfl_xor(__double2loint(total), i, 32);
              total += __hiloint2double(hi_part,lo_part); 
            }
            if (laneid == 0) {
              double *local_fluxm = (double*)(fluxm_ptr + fluxm_offsets * local_point);
              local_fluxm += (l-1);
              atomicAdd(local_fluxm, total);
            }
          }
        }
      }
    }
  }
}

template<int THR_ANGLES>
__global__
void gpu_time_independent_sweep_with_fixup(const Point<3> origin, 
                                         const MomentQuad *qtot_ptr,
                                               double     *flux_ptr,
                                               MomentQuad *fluxm_ptr,
                                         const double     *dinv_ptr,
                                         const double     *t_xs_ptr,
                                               double     *ghostx_out_ptr,
                                               double     *ghosty_out_ptr,
                                               double     *ghostz_out_ptr,
                                         const double     *qim_ptr,
                                         const double     *ghostx_in_ptr,
                                         const double     *ghosty_in_ptr,
                                         const double     *ghostz_in_ptr,
                                         const ByteOffset qtot_offsets[3],
                                         const ByteOffset flux_offsets[3],
                                         const ByteOffset fluxm_offsets[3],
                                         const ByteOffset dinv_offsets[3],
                                         const ByteOffset t_xs_offsets[3],
                                         const ByteOffset ghostx_out_offsets[3],
                                         const ByteOffset ghosty_out_offsets[3],
                                         const ByteOffset ghostz_out_offsets[3],
                                         const ByteOffset qim_offsets[3],
                                         const ByteOffset ghostx_in_offsets[3],
                                         const ByteOffset ghosty_in_offsets[3],
                                         const ByteOffset ghostz_in_offsets[3],
                                         const int x_range, const int y_range, 
                                         const int z_range, const int corner,
                                         const bool stride_x_positive,
                                         const bool stride_y_positive,
                                         const bool stride_z_positive,
                                         const bool mms_source, 
                                         const int num_moments, 
                                         const double hi, const double hj,
                                         const double hk, 
                                         double *yflux_pencil,
                                         double *zflux_plane)
{
  __shared__ int int_trampoline[32];
  __shared__ double double_trampoline[32];

  double psi[THR_ANGLES];
  double pc[THR_ANGLES];
  double psii[THR_ANGLES];
  double psij[THR_ANGLES];
  double psik[THR_ANGLES];
  double hv_x[THR_ANGLES];
  double hv_y[THR_ANGLES];
  double hv_z[THR_ANGLES];
  double fx_hv_x[THR_ANGLES];
  double fx_hv_y[THR_ANGLES];
  double fx_hv_z[THR_ANGLES];

  const int num_angles = THR_ANGLES * blockDim.x;
  const int corner_offset = corner * num_angles * num_moments; 

  unsigned laneid;
  asm volatile("mov.u32 %0, %laneid;" : "=r"(laneid) : );
  unsigned warpid;
  asm volatile("mov.u32 %0, %warpid;" : "=r"(warpid) : );

  const double tolr = 1.0e-12;

  for (int z = 0; z < z_range; z++) {
    for (int y = 0; y < y_range; y++) {
      for (int x = 0; x < x_range; x++) {
        // Figure out the local point that we are working on    
        Point<3> local_point = origin;
        if (stride_x_positive)
          local_point.x[0] += x;
        else
          local_point.x[0] -= x;
        if (stride_y_positive)
          local_point.x[1] += y;
        else
          local_point.x[1] -= y;
        if (stride_z_positive)
          local_point.x[2] += z;
        else
          local_point.x[2] -= z;

        // Compute the angular source
        MomentQuad quad = *(qtot_ptr + qtot_offsets * local_point);
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          psi[ang] = quad[0];
        if (num_moments > 1) {
          for (int l = 1; l < num_moments; l++) {
            const int moment_offset = corner_offset + l * num_angles;
            #pragma unroll
            for (int ang = 0; ang < THR_ANGLES; ang++) {
              psi[ang] += device_ec[moment_offset+ang*blockDim.x+threadIdx.x] * quad[l];
            }
          }
        }

        // If we're doing MMS
        if (mms_source) {
          #pragma unroll
          for (int ang = 0; ang < THR_ANGLES; ang++)
            psi[ang] += *(qim_ptr + ang*blockDim.x + threadIdx.x);
        }

        // Compute the initial solution
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          pc[ang] = psi[ang];
        // X ghost cells
        if (stride_x_positive) {
          // reading from x-1 
          Point<3> ghost_point = local_point;
          ghost_point.x[0] -= 1;
          if (x == 0) {
            // Ghost cell array
            #pragma unroll 
            for (int ang = 0; ang < THR_ANGLES; ang++)
              psii[ang] = *((ghostx_in_ptr + ghostx_in_offsets * ghost_point) + 
                            ang*blockDim.x + threadIdx.x);
          } 
          // Else nothing: psii already contains next flux
        } else {
          // reading from x+1
          Point<3> ghost_point = local_point;
          ghost_point.x[0] += 1;
          // Local coordinates here
          if (x == 0) {
            // Ghost cell array
            #pragma unroll
            for (int ang = 0; ang < THR_ANGLES; ang++)
              psii[ang] = *((ghostx_in_ptr + ghostx_in_offsets * ghost_point) +
                            ang*blockDim.x + threadIdx.x);
          }
          // Else nothing: psii already contains next flux
        }
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          pc[ang] += psii[ang] * device_mu[ang*blockDim.x + threadIdx.x] * hi;
        // Y ghost cells
        if (stride_y_positive) {
          // reading from y-1
          Point<3> ghost_point = local_point;
          ghost_point.x[1] -= 1;
          if (y == 0) {
            // Ghost cell array
            #pragma unroll
            for (int ang = 0; ang < THR_ANGLES; ang++)
              psij[ang] = *((ghosty_in_ptr + ghosty_in_offsets * ghost_point) +
                            ang*blockDim.x + threadIdx.x);
          } else {
            // Local array
            #pragma unroll
            for (int ang = 0; ang < THR_ANGLES; ang++)
              psij[ang] = yflux_pencil[x * num_angles + ang * blockDim.x + threadIdx.x];
          }
        } else {
          // reading from y+1
          Point<3> ghost_point = local_point;
          ghost_point.x[1] += 1;
          // Local coordinates here
          if (y == 0) {
            // Ghost cell array
            #pragma unroll
            for (int ang = 0; ang < THR_ANGLES; ang++)
              psij[ang] = *((ghosty_in_ptr + ghosty_in_offsets * ghost_point) +
                            ang*blockDim.x + threadIdx.x);
          } else {
            // Local array
            #pragma unroll
            for (int ang = 0; ang < THR_ANGLES; ang++)
              psij[ang] = yflux_pencil[x * num_angles + ang * blockDim.x + threadIdx.x];
          }
        }
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          pc[ang] += psij[ang] * device_eta[ang * blockDim.x + threadIdx.x] * hj;
        // Z ghost cells
        if (stride_z_positive) {
          // reading from z-1
          Point<3> ghost_point = local_point;
          ghost_point.x[2] -= 1;
          if (z == 0) {
            // Ghost cell array
            #pragma unroll
            for (int ang = 0; ang < THR_ANGLES; ang++)
              psik[ang] = *((ghostz_in_ptr + ghostz_in_offsets * ghost_point) +
                            ang*blockDim.x + threadIdx.x);
          } else {
            // Local array
            const int offset = (y * x_range + x) * num_angles;
            #pragma unroll
            for (int ang = 0; ang < THR_ANGLES; ang++)
              psik[ang] = zflux_plane[offset + ang * blockDim.x + threadIdx.x];
          }
        } else {
          // reading from z+1
          Point<3> ghost_point = local_point;
          ghost_point.x[2] += 1;
          // Local coordinates here
          if (z == 0) {
            // Ghost cell array
            #pragma unroll
            for (int ang = 0; ang < THR_ANGLES; ang++)
              psik[ang] = *((ghostz_in_ptr + ghostz_in_offsets * ghost_point) + 
                            ang*blockDim.x + threadIdx.x);
          } else {
            // Local array
            const int offset = (y * x_range + x) * num_angles;
            #pragma unroll
            for (int ang = 0; ang < THR_ANGLES; ang++)
              psik[ang] = zflux_plane[offset + ang * blockDim.x + threadIdx.x];
          }
        }
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          pc[ang] += psik[ang] * device_xi[ang * blockDim.x + threadIdx.x] * hk;

        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++) {
          double dinv = *((dinv_ptr + dinv_offsets * local_point) + 
                          ang * blockDim.x + threadIdx.x);
          pc[ang] *= dinv;
        }

        // DO THE FIXUP
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          hv_x[ang] = 1.0;
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          hv_y[ang] = 1.0;
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          hv_z[ang] = 1.0;

        const double t_xs = *(t_xs_ptr + t_xs_offsets * local_point);
        int old_negative_fluxes = 0;
        while (true) {
          unsigned negative_fluxes = 0;
          // Figure out how many negative fluxes we have
          #pragma unroll
          for (int ang = 0; ang < THR_ANGLES; ang++) {
            fx_hv_x[ang] = 2.0 * pc[ang] - psii[ang];
            if (fx_hv_x[ang] < 0.0) {
              hv_x[ang] = 0.0;
              negative_fluxes++;
            }
          }
          #pragma unroll
          for (int ang = 0; ang < THR_ANGLES; ang++) {
            fx_hv_y[ang] = 2.0 * pc[ang] - psij[ang];
            if (fx_hv_y[ang] < 0.0) {
              hv_y[ang] = 0.0;
              negative_fluxes++;
            }
          }
          #pragma unroll
          for (int ang = 0; ang < THR_ANGLES; ang++) {
            fx_hv_z[ang] = 2.0 * pc[ang] - psik[ang];
            if (fx_hv_z[ang] < 0.0) {
              hv_z[ang] = 0.0;
              negative_fluxes++;
            }
          }
          // CTA-wide reduction
          #pragma unroll
          for (int i = 16; i >= 1; i /= 2)
            negative_fluxes += __shfl_xor(negative_fluxes, i, 32);
          // Initialize
          if (warpid == 0)
            int_trampoline[laneid] = 0;
          __syncthreads();
          if (laneid == 0)
            int_trampoline[warpid] = negative_fluxes;
          __syncthreads();
          negative_fluxes = int_trampoline[laneid];
          #pragma unroll
          for (int i = 16; i >= 1; i /= 2)
            negative_fluxes += __shfl_xor(negative_fluxes, i, 32);
          // All threads have the same negative flux count now
          if (negative_fluxes == old_negative_fluxes)
            break;
          old_negative_fluxes = negative_fluxes;
          #pragma unroll
          for (int ang = 0; ang < THR_ANGLES; ang++) {
            pc[ang] = psi[ang] + 0.5 * (
              psii[ang] * device_mu[ang*blockDim.x + threadIdx.x] * hi * (1.0 + hv_x[ang]) +
              psij[ang] * device_eta[ang*blockDim.x + threadIdx.x] * hj * (1.0 + hv_y[ang]) +
              psik[ang] * device_xi[ang*blockDim.x + threadIdx.x] * hk * (1.0 + hv_z[ang]) );
            double den = (pc[ang] <= 0.0) ? 0.0 : (t_xs + 
                device_mu[ang*blockDim.x + threadIdx.x] * hi * hv_x[ang] + 
                device_eta[ang*blockDim.x + threadIdx.x] * hj * hv_y[ang] + 
                device_xi[ang*blockDim.x + threadIdx.x] * hk * hv_z[ang]);
            if (den < tolr)
              pc[ang] = 0.0;
            else
              pc[ang] /= den;
          }
        }
        // Fixup done so compute the update values
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          psii[ang] = fx_hv_x[ang] * hv_x[ang];
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          psij[ang] = fx_hv_y[ang] * hv_y[ang];
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          psik[ang] = fx_hv_z[ang] * hv_z[ang];
        // Write out the ghost regions
        // X ghost
        if (x == (x_range - 1)) {
          #pragma unroll
          for (int ang = 0; ang < THR_ANGLES; ang++)
            *((ghostx_out_ptr + ghostx_out_offsets * local_point) +
                ang * blockDim.x + threadIdx.x) = psii[ang];
        }
        // Y ghost
        if (y == (y_range - 1)) {
          #pragma unroll
          for (int ang = 0; ang < THR_ANGLES; ang++)
            *((ghosty_out_ptr + ghosty_out_offsets * local_point) + 
                ang * blockDim.x + threadIdx.x) = psij[ang];
        } else {
          // Write to the pencil
          const int offset = x * num_angles;
          #pragma unroll
          for (int ang = 0; ang < THR_ANGLES; ang++)
            yflux_pencil[offset + ang * blockDim.x + threadIdx.x] = psij[ang]; 
        }
        // Z ghost
        if (z == (z_range - 1)) {
          #pragma unroll
          for (int ang = 0; ang < THR_ANGLES; ang++)
            *((ghostz_out_ptr + ghostz_out_offsets * local_point) + 
                ang * blockDim.x + threadIdx.x) = psik[ang];
        } else {
          const int offset = (y * x_range + x) * num_angles;
          #pragma unroll
          for (int ang = 0; ang < THR_ANGLES; ang++)
            zflux_plane[offset + ang * blockDim.x + threadIdx.x] = psik[ang];
        }
        // Finally we apply reductions to the flux moments
        double total = 0.0;  
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++) {
          psi[ang] = device_w[ang * blockDim.x + threadIdx.x] * pc[ang];
          total += psi[ang];
        }
        // CTA-wide reduction to one warp and then down to one thread
        #pragma unroll
        for (int i = 16; i >= 1; i /= 2) {
          int hi_part = __shfl_xor(__double2hiint(total), i, 32);
          int lo_part = __shfl_xor(__double2loint(total), i, 32);
          total += __hiloint2double(hi_part,lo_part); 
        }
        if (warpid == 0)
          double_trampoline[laneid] = 0.0;
        __syncthreads();
        if (laneid == 0)
          double_trampoline[warpid] = total;
        __syncthreads();
        if (warpid == 0) {
          total = double_trampoline[laneid];
          #pragma unroll
          for (int i = 16; i >= 1; i /= 2) {
            int hi_part = __shfl_xor(__double2hiint(total), i, 32);
            int lo_part = __shfl_xor(__double2loint(total), i, 32);
            total += __hiloint2double(hi_part,lo_part); 
          }
          // Do the reduction
          if (laneid == 0) {
            double *local_flux = flux_ptr + flux_offsets * local_point;
            atomicAdd(local_flux, total);
          }
        }
        if (num_moments > 1) {
          for (int l = 1; l < num_moments; l++) {
            const int offset = l * num_angles + corner * num_angles * num_moments;
            total = 0.0;
            #pragma unroll
            for (int ang = 0; ang < THR_ANGLES; ang++)
              total += device_ec[offset + ang] * psi[ang];
            __syncthreads();
            if (warpid == 0)
              double_trampoline[laneid] = 0.0;
            __syncthreads();
            if (laneid == 0)
              double_trampoline[warpid] = total;
            __syncthreads();
            if (warpid == 0)
              total = double_trampoline[laneid];
            #pragma unroll
            for (int i = 16; i >= 1; i /= 2) {
              int hi_part = __shfl_xor(__double2hiint(total), i, 32);
              int lo_part = __shfl_xor(__double2loint(total), i, 32);
              total += __hiloint2double(hi_part,lo_part); 
            }
            if (laneid == 0) {
              double *local_fluxm = (double*)(fluxm_ptr + fluxm_offsets * local_point);
              local_fluxm += (l-1);
              atomicAdd(local_fluxm, total);
            }
          }
        }
      }
    }
  }
}

template<int THR_ANGLES>
__global__
void gpu_time_independent_sweep_without_fixup(const Point<3> origin, 
                                         const MomentQuad *qtot_ptr,
                                               double     *flux_ptr,
                                               MomentQuad *fluxm_ptr,
                                         const double     *dinv_ptr,
                                         const double     *t_xs_ptr,
                                               double     *ghostx_out_ptr,
                                               double     *ghosty_out_ptr,
                                               double     *ghostz_out_ptr,
                                         const double     *qim_ptr,
                                         const double     *ghostx_in_ptr,
                                         const double     *ghosty_in_ptr,
                                         const double     *ghostz_in_ptr,
                                         const ByteOffset qtot_offsets[3],
                                         const ByteOffset flux_offsets[3],
                                         const ByteOffset fluxm_offsets[3],
                                         const ByteOffset dinv_offsets[3],
                                         const ByteOffset t_xs_offsets[3],
                                         const ByteOffset ghostx_out_offsets[3],
                                         const ByteOffset ghosty_out_offsets[3],
                                         const ByteOffset ghostz_out_offsets[3],
                                         const ByteOffset qim_offsets[3],
                                         const ByteOffset ghostx_in_offsets[3],
                                         const ByteOffset ghosty_in_offsets[3],
                                         const ByteOffset ghostz_in_offsets[3],
                                         const int x_range, const int y_range, 
                                         const int z_range, const int corner,
                                         const bool stride_x_positive,
                                         const bool stride_y_positive,
                                         const bool stride_z_positive,
                                         const bool mms_source, 
                                         const int num_moments, 
                                         const double hi, const double hj,
                                         const double hk, 
                                         double *yflux_pencil,
                                         double *zflux_plane)
{
  __shared__ double double_trampoline[32];

  double psi[THR_ANGLES];
  double pc[THR_ANGLES];
  double psii[THR_ANGLES];
  double psij[THR_ANGLES];
  double psik[THR_ANGLES];

  const int num_angles = THR_ANGLES * blockDim.x;
  const int corner_offset = corner * num_angles * num_moments; 

  unsigned laneid;
  asm volatile("mov.u32 %0, %laneid;" : "=r"(laneid) : );
  unsigned warpid;
  asm volatile("mov.u32 %0, %warpid;" : "=r"(warpid) : );

  for (int z = 0; z < z_range; z++) {
    for (int y = 0; y < y_range; y++) {
      for (int x = 0; x < x_range; x++) {
        // Figure out the local point that we are working on    
        Point<3> local_point = origin;
        if (stride_x_positive)
          local_point.x[0] += x;
        else
          local_point.x[0] -= x;
        if (stride_y_positive)
          local_point.x[1] += y;
        else
          local_point.x[1] -= y;
        if (stride_z_positive)
          local_point.x[2] += z;
        else
          local_point.x[2] -= z;

        // Compute the angular source
        MomentQuad quad = *(qtot_ptr + qtot_offsets * local_point);
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          psi[ang] = quad[0];
        if (num_moments > 1) {
          for (int l = 1; l < num_moments; l++) {
            const int moment_offset = corner_offset + l * num_angles;
            #pragma unroll
            for (int ang = 0; ang < THR_ANGLES; ang++) {
              psi[ang] += device_ec[moment_offset+ang*blockDim.x+threadIdx.x] * quad[l];
            }
          }
        }

        // If we're doing MMS
        if (mms_source) {
          #pragma unroll
          for (int ang = 0; ang < THR_ANGLES; ang++)
            psi[ang] += *(qim_ptr + ang*blockDim.x + threadIdx.x);
        }

        // Compute the initial solution
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          pc[ang] = psi[ang];
        // X ghost cells
        if (stride_x_positive) {
          // reading from x-1 
          Point<3> ghost_point = local_point;
          ghost_point.x[0] -= 1;
          if (x == 0) {
            // Ghost cell array
            #pragma unroll 
            for (int ang = 0; ang < THR_ANGLES; ang++)
              psii[ang] = *((ghostx_in_ptr + ghostx_in_offsets * ghost_point) + 
                            ang*blockDim.x + threadIdx.x);
          } 
          // Else nothing: psii already contains next flux
        } else {
          // reading from x+1
          Point<3> ghost_point = local_point;
          ghost_point.x[0] += 1;
          // Local coordinates here
          if (x == 0) {
            // Ghost cell array
            #pragma unroll
            for (int ang = 0; ang < THR_ANGLES; ang++)
              psii[ang] = *((ghostx_in_ptr + ghostx_in_offsets * ghost_point) +
                            ang*blockDim.x + threadIdx.x);
          }
          // Else nothing: psii already contains next flux
        }
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          pc[ang] += psii[ang] * device_mu[ang*blockDim.x + threadIdx.x] * hi;
        // Y ghost cells
        if (stride_y_positive) {
          // reading from y-1
          Point<3> ghost_point = local_point;
          ghost_point.x[1] -= 1;
          if (y == 0) {
            // Ghost cell array
            #pragma unroll
            for (int ang = 0; ang < THR_ANGLES; ang++)
              psij[ang] = *((ghosty_in_ptr + ghosty_in_offsets * ghost_point) +
                            ang*blockDim.x + threadIdx.x);
          } else {
            // Local array
            #pragma unroll
            for (int ang = 0; ang < THR_ANGLES; ang++)
              psij[ang] = yflux_pencil[x * num_angles + ang * blockDim.x + threadIdx.x];
          }
        } else {
          // reading from y+1
          Point<3> ghost_point = local_point;
          ghost_point.x[1] += 1;
          // Local coordinates here
          if (y == 0) {
            // Ghost cell array
            #pragma unroll
            for (int ang = 0; ang < THR_ANGLES; ang++)
              psij[ang] = *((ghosty_in_ptr + ghosty_in_offsets * ghost_point) +
                            ang*blockDim.x + threadIdx.x);
          } else {
            // Local array
            #pragma unroll
            for (int ang = 0; ang < THR_ANGLES; ang++)
              psij[ang] = yflux_pencil[x * num_angles + ang * blockDim.x + threadIdx.x];
          }
        }
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          pc[ang] += psij[ang] * device_eta[ang * blockDim.x + threadIdx.x] * hj;
        // Z ghost cells
        if (stride_z_positive) {
          // reading from z-1
          Point<3> ghost_point = local_point;
          ghost_point.x[2] -= 1;
          if (z == 0) {
            // Ghost cell array
            #pragma unroll
            for (int ang = 0; ang < THR_ANGLES; ang++)
              psik[ang] = *((ghostz_in_ptr + ghostz_in_offsets * ghost_point) +
                            ang*blockDim.x + threadIdx.x);
          } else {
            // Local array
            const int offset = (y * x_range + x) * num_angles;
            #pragma unroll
            for (int ang = 0; ang < THR_ANGLES; ang++)
              psik[ang] = zflux_plane[offset + ang * blockDim.x + threadIdx.x];
          }
        } else {
          // reading from z+1
          Point<3> ghost_point = local_point;
          ghost_point.x[2] += 1;
          // Local coordinates here
          if (z == 0) {
            // Ghost cell array
            #pragma unroll
            for (int ang = 0; ang < THR_ANGLES; ang++)
              psik[ang] = *((ghostz_in_ptr + ghostz_in_offsets * ghost_point) + 
                            ang*blockDim.x + threadIdx.x);
          } else {
            // Local array
            const int offset = (y * x_range + x) * num_angles;
            #pragma unroll
            for (int ang = 0; ang < THR_ANGLES; ang++)
              psik[ang] = zflux_plane[offset + ang * blockDim.x + threadIdx.x];
          }
        }
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          pc[ang] += psik[ang] * device_xi[ang * blockDim.x + threadIdx.x] * hk;

        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++) {
          double dinv = *((dinv_ptr + dinv_offsets * local_point) + 
                          ang * blockDim.x + threadIdx.x);
          pc[ang] *= dinv;
        }

        // NO FIXUP
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          psii[ang] = 2.0 * pc[ang] - psii[ang];
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          psij[ang] = 2.0 * pc[ang] - psij[ang];
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          psik[ang] = 2.0 * pc[ang] - psik[ang];

        // Write out the ghost regions
        // X ghost
        if (x == (x_range - 1)) {
          #pragma unroll
          for (int ang = 0; ang < THR_ANGLES; ang++)
            *((ghostx_out_ptr + ghostx_out_offsets * local_point) +
                ang * blockDim.x + threadIdx.x) = psii[ang];
        }
        // Y ghost
        if (y == (y_range - 1)) {
          #pragma unroll
          for (int ang = 0; ang < THR_ANGLES; ang++)
            *((ghosty_out_ptr + ghosty_out_offsets * local_point) + 
                ang * blockDim.x + threadIdx.x) = psij[ang];
        } else {
          // Write to the pencil
          const int offset = x * num_angles;
          #pragma unroll
          for (int ang = 0; ang < THR_ANGLES; ang++)
            yflux_pencil[offset + ang * blockDim.x + threadIdx.x] = psij[ang]; 
        }
        // Z ghost
        if (z == (z_range - 1)) {
          #pragma unroll
          for (int ang = 0; ang < THR_ANGLES; ang++)
            *((ghostz_out_ptr + ghostz_out_offsets * local_point) + 
                ang * blockDim.x + threadIdx.x) = psik[ang];
        } else {
          const int offset = (y * x_range + x) * num_angles;
          #pragma unroll
          for (int ang = 0; ang < THR_ANGLES; ang++)
            zflux_plane[offset + ang * blockDim.x + threadIdx.x] = psik[ang];
        }
        // Finally we apply reductions to the flux moments
        double total = 0.0;  
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++) {
          psi[ang] = device_w[ang * blockDim.x + threadIdx.x] * pc[ang];
          total += psi[ang];
        }
        // CTA-wide reduction to one warp and then down to one thread
        #pragma unroll
        for (int i = 16; i >= 1; i /= 2) {
          int hi_part = __shfl_xor(__double2hiint(total), i, 32);
          int lo_part = __shfl_xor(__double2loint(total), i, 32);
          total += __hiloint2double(hi_part,lo_part); 
        }
        if (warpid == 0)
          double_trampoline[laneid] = 0.0;
        __syncthreads();
        if (laneid == 0)
          double_trampoline[warpid] = total;
        __syncthreads();
        if (warpid == 0) {
          total = double_trampoline[laneid];
          #pragma unroll
          for (int i = 16; i >= 1; i /= 2) {
            int hi_part = __shfl_xor(__double2hiint(total), i, 32);
            int lo_part = __shfl_xor(__double2loint(total), i, 32);
            total += __hiloint2double(hi_part,lo_part); 
          }
          // Do the reduction
          if (laneid == 0) {
            double *local_flux = flux_ptr + flux_offsets * local_point;
            atomicAdd(local_flux, total);
          }
        }
        if (num_moments > 1) {
          for (int l = 1; l < num_moments; l++) {
            const int offset = l * num_angles + corner * num_angles * num_moments;
            total = 0.0;
            #pragma unroll
            for (int ang = 0; ang < THR_ANGLES; ang++)
              total += device_ec[offset + ang] * psi[ang];
            __syncthreads();
            if (warpid == 0)
              double_trampoline[laneid] = 0.0;
            __syncthreads();
            if (laneid == 0)
              double_trampoline[warpid] = total;
            __syncthreads();
            if (warpid == 0)
              total = double_trampoline[laneid];
            #pragma unroll
            for (int i = 16; i >= 1; i /= 2) {
              int hi_part = __shfl_xor(__double2hiint(total), i, 32);
              int lo_part = __shfl_xor(__double2loint(total), i, 32);
              total += __hiloint2double(hi_part,lo_part); 
            }
            if (laneid == 0) {
              double *local_fluxm = (double*)(fluxm_ptr + fluxm_offsets * local_point);
              local_fluxm += (l-1);
              atomicAdd(local_fluxm, total);
            }
          }
        }
      }
    }
  }
}

