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

#include <cstdio>

#include "snap.h"
#include "snap_cuda_help.h"

#ifndef MAX_GPUS
#define MAX_GPUS 16
#endif
#ifndef MAX_OCTANTS
#define MAX_OCTANTS  8
#endif
#ifndef MAX_STREAMS
#define MAX_STREAMS   12
#endif

static double *ec_d[MAX_GPUS]; /*corners * moments * angles*/
static double *mu_d[MAX_GPUS]; /*angles*/
static double *eta_d[MAX_GPUS]; /*angles*/
static double *xi_d[MAX_GPUS]; /*angles*/
static double *w_d[MAX_GPUS]; /*angles*/

static double *flux_x_d[MAX_GPUS][MAX_STREAMS]; /* ny * nz * angles */
static double *flux_y_d[MAX_GPUS][MAX_STREAMS]; /* nx * nz * angles */
static double *flux_z_d[MAX_GPUS][MAX_STREAMS]; /* nx * ny * angles */
static int *mutex_in_d[MAX_GPUS][MAX_STREAMS];
static int *mutex_out_d[MAX_GPUS][MAX_STREAMS];
static cudaStream_t flux_streams[MAX_GPUS][MAX_STREAMS];

static int blocks_per_sweep[MAX_GPUS];
static int total_wavefronts[MAX_GPUS];
static int max_wavefront_length[MAX_GPUS];

static int *wavefront_length_d[MAX_GPUS][MAX_OCTANTS];
static int *wavefront_offset_d[MAX_GPUS][MAX_OCTANTS];
static int *wavefront_x_d[MAX_GPUS][MAX_OCTANTS];
static int *wavefront_y_d[MAX_GPUS][MAX_OCTANTS];
static int *wavefront_z_d[MAX_GPUS][MAX_OCTANTS];


__host__
static bool contains_point(Point<3> &point, int xlo, int xhi, 
                           int ylo, int yhi, int zlo, int zhi)
{
  if ((point[0] < xlo) || (point[0] > xhi))
    return false;
  if ((point[1] < ylo) || (point[1] > yhi))
    return false;
  if ((point[2] < zlo) || (point[2] > zhi))
    return false;
  return true;
}

__host__
void initialize_gpu_context(const double *ec_h, const double *mu_h,
                            const double *eta_h, const double *xi_h,
                            const double *w_h, const int num_angles,
                            const int num_moments, const int num_octants,
                            const int nx_per_chunk, const int ny_per_chunk,
                            const int nz_per_chunk)
{
  int gpu;
  cudaGetDevice(&gpu);
  assert(gpu < MAX_GPUS);
  assert(num_octants <= MAX_OCTANTS);
  const size_t ec_size = num_angles * num_moments * num_octants * sizeof(double);
  if (cudaMalloc((void**)&ec_d[gpu], ec_size) != cudaSuccess)
  {
    printf("ERROR: out of memory for ec_d of %zd bytes on GPU %d\n", ec_size, gpu);
    exit(1);
  }
  cudaMemcpy(ec_d[gpu], ec_h, ec_size, cudaMemcpyHostToDevice);

  const size_t angle_size = num_angles * sizeof(double);
  if (cudaMalloc((void**)&mu_d[gpu], angle_size) != cudaSuccess)
  {
    printf("ERROR: out of memory for mu_d of %zd bytes on GPU %d\n", angle_size, gpu);
    exit(1);
  }
  cudaMemcpy(mu_d[gpu], mu_h, angle_size, cudaMemcpyHostToDevice);

  if (cudaMalloc((void**)&eta_d[gpu], angle_size) != cudaSuccess)
  {
    printf("ERROR: out of memory for eta_d of %zd bytes on GPU %d\n", angle_size, gpu);
    exit(1);
  }
  cudaMemcpy(eta_d[gpu], eta_h, angle_size, cudaMemcpyHostToDevice);

  if (cudaMalloc((void**)&xi_d[gpu], angle_size) != cudaSuccess)
  {
    printf("ERROR: out of memory for xi_d of %zd bytes on GPU %d\n", angle_size, gpu);
    exit(1);
  }
  cudaMemcpy(xi_d[gpu], xi_h, angle_size, cudaMemcpyHostToDevice);

  if (cudaMalloc((void**)&w_d[gpu], angle_size) != cudaSuccess)
  {
    printf("ERROR: out of memory for w_d of %zd bytes on GPU %d\n", angle_size, gpu);
    exit(1);
  }
  cudaMemcpy(w_d[gpu], w_h, angle_size, cudaMemcpyHostToDevice);

  const size_t flux_x_size = ny_per_chunk * nz_per_chunk * angle_size;
  for (int idx = 0; idx < MAX_STREAMS; idx++)
    if (cudaMalloc((void**)&flux_x_d[gpu][idx], flux_x_size) != cudaSuccess)
    {
      printf("ERROR: out of memory for flux X %d of %zd bytes on GPU %d\n",
             idx, flux_x_size, gpu);
      exit(1);
    }

  const size_t flux_y_size = nx_per_chunk * nz_per_chunk * angle_size;
  for (int idx = 0; idx < MAX_STREAMS; idx++)
    if (cudaMalloc((void**)&flux_y_d[gpu][idx], flux_y_size) != cudaSuccess)
    {
      printf("ERROR: out of memory for flux Y %d of %zd bytes on GPU %d\n",
             idx, flux_y_size, gpu);
      exit(1);
    }

  const size_t flux_z_size = nx_per_chunk * ny_per_chunk * angle_size;
  for (int idx = 0; idx < MAX_STREAMS; idx++)
    if (cudaMalloc((void**)&flux_z_d[gpu][idx], flux_z_size) != cudaSuccess)
    {
      printf("ERROR: out of memory for flux Z %d of %zd bytes on GPU %d\n",
             idx, flux_z_size, gpu);
      exit(1);
    }
  for (int idx = 0; idx < MAX_STREAMS; idx++)
    flux_streams[gpu][idx] = 0;
   
  // Initialize this to zero so we can compute it later
  blocks_per_sweep[gpu] = 0;
  const long long zeroes[3] = { 0, 0, 0 };
  const long long chunks[3] = { nx_per_chunk, ny_per_chunk, nz_per_chunk };
  // Compute the mapping from corners to wavefronts
  for (int corner = 0; corner < num_octants; corner++)
  {
    Point<3> strides[3] = 
      { Point<3>(zeroes), Point<3>(zeroes), Point<3>(zeroes) };
    Point<3> start;
    for (int i = 0; i < 3; i++)
    {
      start[i] = ((corner & (0x1 << i)) ? 0 : chunks[i]-1);
      strides[i][i] = ((corner & (0x1 << i)) ? 1 : -1);
    }
    std::set<Point<3> > current_points;
    current_points.insert(Point<3>(start));
    // Do a little BFS to handle weird rectangle shapes correctly
    unsigned wavefront_count = 0;
    std::vector<std::vector<Point<3> > > wavefront_map;
    int max_length = 0;
    while (!current_points.empty())
    {
      wavefront_map.push_back(std::vector<Point<3> >());
      std::vector<Point<3> > &wavefront_points = wavefront_map[wavefront_count];
      std::set<Point<3> > next_points;
      for (std::set<Point<3> >::const_iterator it = 
            current_points.begin(); it != current_points.end(); it++)
      {
        // Save the point in this wavefront
        wavefront_points.push_back(*it);
        for (int i = 0; i < 3; i++)
        {
          Point<3> next = *it + strides[i];
          if (contains_point(next, 0, nx_per_chunk-1, 0, 
                             ny_per_chunk-1, 0, nz_per_chunk-1))
            next_points.insert(Point<3>(next));
        }
      }
      current_points = next_points;
      wavefront_count++;
      if (wavefront_points.size() > max_length)
        max_length = wavefront_points.size();
    }
    if (corner == 0)
    {
      total_wavefronts[gpu] = wavefront_count;
      max_wavefront_length[gpu] = max_length;
    }
    else
    {
      assert(wavefront_count == total_wavefronts[gpu]);
      assert(max_length == max_wavefront_length[gpu]);
    }
    int *wavefront_length_h = (int*)malloc(wavefront_count * sizeof(int));
    int *wavefront_offset_h = (int*)malloc(wavefront_count* sizeof(int));
    int offset = 0;
    for (int i = 0; i < wavefront_count; i++)
    {
      wavefront_offset_h[i] = offset;
      wavefront_length_h[i] = wavefront_map[i].size();
      offset += wavefront_map[i].size();
    }
    if (cudaMalloc((void**)&wavefront_length_d[gpu][corner], 
          wavefront_count * sizeof(int)) != cudaSuccess)
    {
      printf("ERROR: out of memory for wavefront length corner %d of %zd bytes on GPU %d\n",
             corner, wavefront_count * sizeof(int), gpu);
      exit(1);
    }
    cudaMemcpy(wavefront_length_d[gpu][corner], wavefront_length_h, 
        wavefront_count * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaMalloc((void**)&wavefront_offset_d[gpu][corner], 
          wavefront_count * sizeof(int)) != cudaSuccess)
    {
      printf("ERROR: out of memory for wavefront offset corner %d of %zd bytes on GPU %d\n",
             corner, wavefront_count * sizeof(int), gpu);
      exit(1);
    }
    cudaMemcpy(wavefront_offset_d[gpu][corner], wavefront_offset_h, 
        wavefront_count * sizeof(int), cudaMemcpyDeviceToHost);
    int *wavefront_x_h = (int*)malloc(offset * sizeof(int));
    int *wavefront_y_h = (int*)malloc(offset * sizeof(int));
    int *wavefront_z_h = (int*)malloc(offset * sizeof(int));
    offset = 0;
    for (int i = 0; i < wavefront_count; i++)
      for (int j = 0; j < wavefront_map[i].size(); j++, offset++)
      {
        wavefront_x_h[offset] = wavefront_map[i][j][0];
        wavefront_y_h[offset] = wavefront_map[i][j][1];
        wavefront_z_h[offset] = wavefront_map[i][j][2];
      }
    if (cudaMalloc((void**)&wavefront_x_d[gpu][corner], offset * sizeof(int)) != cudaSuccess)
    {
      printf("ERROR: out of memory for wavefront x of corner %d of %zd bytes on GPU %d\n",
          corner, offset * sizeof(int), gpu);
      exit(1);
    }
    cudaMemcpy(wavefront_x_d[gpu][corner], wavefront_x_h, 
        offset * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaMalloc((void**)&wavefront_y_d[gpu][corner], offset * sizeof(int)) != cudaSuccess)
    {
      printf("ERROR: out of memory for wavefront y of corner %d of %zd bytes on GPU %d\n",
          corner, offset * sizeof(int), gpu);
      exit(1);
    }
    cudaMemcpy(wavefront_y_d[gpu][corner], wavefront_y_h, 
        offset * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaMalloc((void**)&wavefront_z_d[gpu][corner], offset * sizeof(int)) != cudaSuccess)
    {
      printf("ERROR: out of memory for wavefront z of corner %d of %zd bytes on GPU %d\n",
          corner, offset * sizeof(int), gpu);
      exit(1);
    }
    cudaMemcpy(wavefront_z_d[gpu][corner], wavefront_z_h, 
        offset * sizeof(int), cudaMemcpyHostToDevice);
    // Make sure all the copies are done before we delete host memory
    cudaDeviceSynchronize();
    free(wavefront_length_h);
    free(wavefront_offset_h);
    free(wavefront_x_h);
    free(wavefront_y_h);
    free(wavefront_z_h);
  }    
}

// This is from expxs but it uses the same constants
template<int GROUPS>
__global__
void gpu_geometry_param(const Point<3> origin,
                        const AccessorArray<GROUPS,
                                AccessorRO<double,3>,3> fa_xs,
                        const AccessorArray<GROUPS,
                                AccessorWO<double,3>,3> fa_dinv,
                        const ConstBuffer<GROUPS,double> vdelt,
                        const double hi, const double hj, const double hk,
                        const int angles_per_thread,
                        const double *device_mu,
                        const double *device_eta,
                        const double *device_xi)

{
  const int x = blockIdx.x;
  const int y = blockIdx.y;
  const int z = blockIdx.z;
  const Point<3> p = origin + Point<3>(x,y,z);
  for (int i = 0; i < angles_per_thread; i++) {
    const int ang = i * blockDim.x + threadIdx.x;

    const double sum = hi * device_mu[ang] + hj * device_eta[ang] + hk * device_xi[ang];
    #pragma unroll
    for (int g = 0; g < GROUPS; g++) {
      const double *xs_ptr = fa_xs[g].ptr(p);
      double xs;
      // Cache this at all levels since it is shared across all threads in the CTA
      asm volatile("ld.global.ca.f64 %0, [%1];" : "=d"(xs) : "l"(xs_ptr) : "memory");
      double result = 1.0 / (xs + vdelt[g] + sum);
      double *dinv_ptr = fa_dinv[g].ptr(p);
      asm volatile("st.global.cs.f64 [%0], %1;" : : 
                    "l"(dinv_ptr+ang), "d"(result) : "memory");
    }
  }
}

__host__
void run_geometry_param(const std::vector<AccessorRO<double,3> > &fa_xs,
                        const std::vector<AccessorWO<double,3> > &fa_dinv,
                        const std::vector<double> &vdelts,
                        const double hi, const double hj, const double hk,
                        const Rect<3> &subgrid_bounds, const int num_angles)
{
  // Figure out the launch bounds, then dispatch
  const int x_range = (subgrid_bounds.hi[0] - subgrid_bounds.lo[0]) + 1; 
  const int y_range = (subgrid_bounds.hi[1] - subgrid_bounds.lo[1]) + 1;
  const int z_range = (subgrid_bounds.hi[2] - subgrid_bounds.lo[2]) + 1;

  const int max_threads_per_cta = 1024;
  const int angles_per_thread = 
    (num_angles + max_threads_per_cta - 1) / max_threads_per_cta;
  // Have to be evenly divisible for now
  assert((num_angles % angles_per_thread) == 0);
  const int threads_per_cta = num_angles / angles_per_thread;
  dim3 block(threads_per_cta, 1, 1);
  dim3 grid(x_range, y_range, z_range);
  // TODO: Replace template foolishness with terra
  assert(fa_xs.size() == fa_dinv.size());
  int gpu;
  cudaGetDevice(&gpu);
  switch (fa_xs.size())
  {
    case 1:
      {
        gpu_geometry_param<1><<<grid,block>>>(subgrid_bounds.lo,
                                      AccessorArray<1,
                                        AccessorRO<double,3>,3>(fa_xs),
                                      AccessorArray<1,
                                        AccessorWO<double,3>,3>(fa_dinv),
                                      ConstBuffer<1,double>(vdelts),
                                      hi, hj, hk, angles_per_thread,
                                      mu_d[gpu], eta_d[gpu], xi_d[gpu]);
        break;
      }
    case 2:
      {
        gpu_geometry_param<2><<<grid,block>>>(subgrid_bounds.lo,
                                      AccessorArray<2,
                                        AccessorRO<double,3>,3>(fa_xs),
                                      AccessorArray<2,
                                        AccessorWO<double,3>,3>(fa_dinv),
                                      ConstBuffer<2,double>(vdelts),
                                      hi, hj, hk, angles_per_thread,
                                      mu_d[gpu], eta_d[gpu], xi_d[gpu]);
        break;
      }
    case 3:
      {
        gpu_geometry_param<3><<<grid,block>>>(subgrid_bounds.lo,
                                      AccessorArray<3,
                                        AccessorRO<double,3>,3>(fa_xs),
                                      AccessorArray<3,
                                        AccessorWO<double,3>,3>(fa_dinv),
                                      ConstBuffer<3,double>(vdelts),
                                      hi, hj, hk, angles_per_thread,
                                      mu_d[gpu], eta_d[gpu], xi_d[gpu]);
        break;
      }
    case 4:
      {
        gpu_geometry_param<4><<<grid,block>>>(subgrid_bounds.lo,
                                      AccessorArray<4,
                                        AccessorRO<double,3>,3>(fa_xs),
                                      AccessorArray<4,
                                        AccessorWO<double,3>,3>(fa_dinv),
                                      ConstBuffer<4,double>(vdelts),
                                      hi, hj, hk, angles_per_thread,
                                      mu_d[gpu], eta_d[gpu], xi_d[gpu]);
        break;
      }
    case 5:
      {
        gpu_geometry_param<5><<<grid,block>>>(subgrid_bounds.lo,
                                      AccessorArray<5,
                                        AccessorRO<double,3>,3>(fa_xs),
                                      AccessorArray<5,
                                        AccessorWO<double,3>,3>(fa_dinv),
                                      ConstBuffer<5,double>(vdelts),
                                      hi, hj, hk, angles_per_thread,
                                      mu_d[gpu], eta_d[gpu], xi_d[gpu]);
        break;
      }
    case 6:
      {
        gpu_geometry_param<6><<<grid,block>>>(subgrid_bounds.lo,
                                      AccessorArray<6,
                                        AccessorRO<double,3>,3>(fa_xs),
                                      AccessorArray<6,
                                        AccessorWO<double,3>,3>(fa_dinv),
                                      ConstBuffer<6,double>(vdelts),
                                      hi, hj, hk, angles_per_thread,
                                      mu_d[gpu], eta_d[gpu], xi_d[gpu]);
        break;
      }
    case 7:
      {
        gpu_geometry_param<7><<<grid,block>>>(subgrid_bounds.lo,
                                      AccessorArray<7,
                                        AccessorRO<double,3>,3>(fa_xs),
                                      AccessorArray<7,
                                        AccessorWO<double,3>,3>(fa_dinv),
                                      ConstBuffer<7,double>(vdelts),
                                      hi, hj, hk, angles_per_thread,
                                      mu_d[gpu], eta_d[gpu], xi_d[gpu]);
        break;
      }
    case 8:
      {
        gpu_geometry_param<8><<<grid,block>>>(subgrid_bounds.lo,
                                      AccessorArray<8,
                                        AccessorRO<double,3>,3>(fa_xs),
                                      AccessorArray<8,
                                        AccessorWO<double,3>,3>(fa_dinv),
                                      ConstBuffer<8,double>(vdelts),
                                      hi, hj, hk, angles_per_thread,
                                      mu_d[gpu], eta_d[gpu], xi_d[gpu]);
        break;
      }
    case 9:
      {
        gpu_geometry_param<9><<<grid,block>>>(subgrid_bounds.lo,
                                      AccessorArray<9,
                                        AccessorRO<double,3>,3>(fa_xs),
                                      AccessorArray<9,
                                        AccessorWO<double,3>,3>(fa_dinv),
                                      ConstBuffer<9,double>(vdelts),
                                      hi, hj, hk, angles_per_thread,
                                      mu_d[gpu], eta_d[gpu], xi_d[gpu]);
        break;
      }
    case 10:
      {
        gpu_geometry_param<10><<<grid,block>>>(subgrid_bounds.lo,
                                      AccessorArray<10,
                                        AccessorRO<double,3>,3>(fa_xs),
                                      AccessorArray<10,
                                        AccessorWO<double,3>,3>(fa_dinv),
                                      ConstBuffer<10,double>(vdelts),
                                      hi, hj, hk, angles_per_thread,
                                      mu_d[gpu], eta_d[gpu], xi_d[gpu]);
        break;
      }
    case 11:
      {
        gpu_geometry_param<11><<<grid,block>>>(subgrid_bounds.lo,
                                      AccessorArray<11,
                                        AccessorRO<double,3>,3>(fa_xs),
                                      AccessorArray<11,
                                        AccessorWO<double,3>,3>(fa_dinv),
                                      ConstBuffer<11,double>(vdelts),
                                      hi, hj, hk, angles_per_thread,
                                      mu_d[gpu], eta_d[gpu], xi_d[gpu]);
        break;
      }
    case 12:
      {
        gpu_geometry_param<12><<<grid,block>>>(subgrid_bounds.lo,
                                      AccessorArray<12,
                                        AccessorRO<double,3>,3>(fa_xs),
                                      AccessorArray<12,
                                        AccessorWO<double,3>,3>(fa_dinv),
                                      ConstBuffer<12,double>(vdelts),
                                      hi, hj, hk, angles_per_thread,
                                      mu_d[gpu], eta_d[gpu], xi_d[gpu]);
        break;
      }
    case 13:
      {
        gpu_geometry_param<13><<<grid,block>>>(subgrid_bounds.lo,
                                      AccessorArray<13,
                                        AccessorRO<double,3>,3>(fa_xs),
                                      AccessorArray<13,
                                        AccessorWO<double,3>,3>(fa_dinv),
                                      ConstBuffer<13,double>(vdelts),
                                      hi, hj, hk, angles_per_thread,
                                      mu_d[gpu], eta_d[gpu], xi_d[gpu]);
        break;
      }
    case 14:
      {
        gpu_geometry_param<14><<<grid,block>>>(subgrid_bounds.lo,
                                      AccessorArray<14,
                                        AccessorRO<double,3>,3>(fa_xs),
                                      AccessorArray<14,
                                        AccessorWO<double,3>,3>(fa_dinv),
                                      ConstBuffer<14,double>(vdelts),
                                      hi, hj, hk, angles_per_thread,
                                      mu_d[gpu], eta_d[gpu], xi_d[gpu]);
        break;
      }
    case 15:
      {
        gpu_geometry_param<15><<<grid,block>>>(subgrid_bounds.lo,
                                      AccessorArray<15,
                                        AccessorRO<double,3>,3>(fa_xs),
                                      AccessorArray<15,
                                        AccessorWO<double,3>,3>(fa_dinv),
                                      ConstBuffer<15,double>(vdelts),
                                      hi, hj, hk, angles_per_thread,
                                      mu_d[gpu], eta_d[gpu], xi_d[gpu]);
        break;
      }
    case 16:
      {
        gpu_geometry_param<16><<<grid,block>>>(subgrid_bounds.lo,
                                      AccessorArray<16,
                                        AccessorRO<double,3>,3>(fa_xs),
                                      AccessorArray<16,
                                        AccessorWO<double,3>,3>(fa_dinv),
                                      ConstBuffer<16,double>(vdelts),
                                      hi, hj, hk, angles_per_thread,
                                      mu_d[gpu], eta_d[gpu], xi_d[gpu]);
        break;
      }
    case 24:
      {
        gpu_geometry_param<24><<<grid,block>>>(subgrid_bounds.lo,
                                      AccessorArray<24,
                                        AccessorRO<double,3>,3>(fa_xs),
                                      AccessorArray<24,
                                        AccessorWO<double,3>,3>(fa_dinv),
                                      ConstBuffer<24,double>(vdelts),
                                      hi, hj, hk, angles_per_thread,
                                      mu_d[gpu], eta_d[gpu], xi_d[gpu]);
        break;
      }
    case 32:
      {
        gpu_geometry_param<32><<<grid,block>>>(subgrid_bounds.lo,
                                      AccessorArray<32,
                                        AccessorRO<double,3>,3>(fa_xs),
                                      AccessorArray<32,
                                        AccessorWO<double,3>,3>(fa_dinv),
                                      ConstBuffer<32,double>(vdelts),
                                      hi, hj, hk, angles_per_thread,
                                      mu_d[gpu], eta_d[gpu], xi_d[gpu]);
        break;
      }
    case 40:
      {
        gpu_geometry_param<40><<<grid,block>>>(subgrid_bounds.lo,
                                      AccessorArray<40,
                                        AccessorRO<double,3>,3>(fa_xs),
                                      AccessorArray<40,
                                        AccessorWO<double,3>,3>(fa_dinv),
                                      ConstBuffer<40,double>(vdelts),
                                      hi, hj, hk, angles_per_thread,
                                      mu_d[gpu], eta_d[gpu], xi_d[gpu]);
        break;
      }
    case 48:
      {
        gpu_geometry_param<48><<<grid,block>>>(subgrid_bounds.lo,
                                      AccessorArray<48,
                                        AccessorRO<double,3>,3>(fa_xs),
                                      AccessorArray<48,
                                        AccessorWO<double,3>,3>(fa_dinv),
                                      ConstBuffer<48,double>(vdelts),
                                      hi, hj, hk, angles_per_thread,
                                      mu_d[gpu], eta_d[gpu], xi_d[gpu]);
        break;
      }
    case 56:
      {
        gpu_geometry_param<56><<<grid,block>>>(subgrid_bounds.lo,
                                      AccessorArray<56,
                                        AccessorRO<double,3>,3>(fa_xs),
                                      AccessorArray<56,
                                        AccessorWO<double,3>,3>(fa_dinv),
                                      ConstBuffer<56,double>(vdelts),
                                      hi, hj, hk, angles_per_thread,
                                      mu_d[gpu], eta_d[gpu], xi_d[gpu]);
        break;
      }
    case 64:
      {
        gpu_geometry_param<64><<<grid,block>>>(subgrid_bounds.lo,
                                      AccessorArray<64,
                                        AccessorRO<double,3>,3>(fa_xs),
                                      AccessorArray<64,
                                        AccessorWO<double,3>,3>(fa_dinv),
                                      ConstBuffer<64,double>(vdelts),
                                      hi, hj, hk, angles_per_thread,
                                      mu_d[gpu], eta_d[gpu], xi_d[gpu]);
        break;
      }
    default:
      exit(1); // need more cases
  }
}

__device__ __forceinline__
void ourAtomicAdd(double *ptr, double value)
{
#if __CUDA_ARCH__ < 600
  unsigned long long int* address_as_ull = (unsigned long long int*)ptr; 
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed;
  do { 
    assumed = old; 
    old = atomicCAS(address_as_ull, assumed, 
        __double_as_longlong(value + __longlong_as_double(assumed))); 
  } while (assumed != old);
#else
  // We have double precision atomicAdd starting in Pascal
  atomicAdd(ptr, value);
#endif
}

template<int DIM>
__device__ __forceinline__
double angle_read(const AccessorRO<double,DIM> &fa_acc,
                  const Point<DIM> &point, int ang)
{
  const double *ptr = fa_acc.ptr(point);
  ptr += ang * blockDim.x + threadIdx.x;
  double result;
  asm volatile("ld.global.cs.f64 %0, [%1];" : "=d"(result) : "l"(ptr) : "memory");
  return result;
}

template<int DIM>
__device__ __forceinline__
double angle_read(const AccessorRW<double,DIM> &fa_acc,
                  const Point<DIM> &point, int ang)
{
  const double *ptr = fa_acc.ptr(point);
  ptr += ang * blockDim.x + threadIdx.x;
  double result;
  asm volatile("ld.global.cs.f64 %0, [%1];" : "=d"(result) : "l"(ptr) : "memory");
  return result;
}

template<int DIM>
__device__ __forceinline__
void angle_write(const AccessorWO<double,DIM> &fa_acc,
                 const Point<DIM> &point, int ang, double val)
{
  double *ptr = fa_acc.ptr(point);
  ptr += ang * blockDim.x + threadIdx.x;
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(ptr), "d"(val) : "memory");
}

template<int DIM>
__device__ __forceinline__
void angle_write(const AccessorRW<double,DIM> &fa_acc,
                 const Point<DIM> &point, int ang, double val)
{
  double *ptr = fa_acc.ptr(point);
  ptr += ang * blockDim.x + threadIdx.x;
  asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(ptr), "d"(val) : "memory");
}

__device__ __forceinline__
double read_flux(volatile double *src, int a, int b, 
                 int pitch, int num_angles, int ang)
{
  return src[(a * pitch + b) * num_angles + ang * blockDim.x + threadIdx.x];
}

__device__ __forceinline__
void write_flux(volatile double *dst, int a, int b,
                int pitch, int num_angles, int ang, double value)
{
  dst[(a * pitch + b) * num_angles + ang * blockDim.x + threadIdx.x] = value;
}

__device__ __forceinline__
Point<2> ghostx_point(const Point<3> &local_point)
{
  Point<2> ghost;
  ghost[0] = local_point[1]; // y
  ghost[1] = local_point[2]; // z
  return ghost;
}

__device__ __forceinline__
Point<2> ghosty_point(const Point<3> &local_point)
{
  Point<2> ghost;
  ghost[0] = local_point[0]; // x
  ghost[1] = local_point[2]; // z
  return ghost;
}

__device__ __forceinline__
Point<2> ghostz_point(const Point<3> &local_point)
{
  Point<2> ghost;
  ghost[0] = local_point[0]; // x
  ghost[1] = local_point[1]; // y
  return ghost;
}

__device__ __forceinline__
void kernel_barrier(int val, volatile int *mutexin, volatile int *mutexout)
{
  __syncthreads();
  __threadfence();

  if (threadIdx.x == 0) {
    mutexin[blockIdx.x] = val;
  }

  if (blockIdx.x == 0) {
    for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x) {
      while (mutexin[i] != val) { }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x) {
      mutexout[i] = val;
    }
  }

  if (threadIdx.x == 0) {
    while(mutexout[blockIdx.x] != val) { }
  }

  __syncthreads();
}

template<int THR_ANGLES>
__global__
void gpu_time_dependent_sweep_with_fixup(const Point<3> origin, 
                                         const AccessorRO<MomentQuad,3> fa_qtot,
                                         const AccessorRW<double,3> fa_flux,
                                         const AccessorRW<MomentTriple,3> fa_fluxm,
                                         const AccessorRO<double,3> fa_dinv,
                                         const AccessorRO<double,3> fa_time_flux_in,
                                         const AccessorWO<double,3> fa_time_flux_out,
                                         const AccessorRO<double,3> fa_t_xs,
                                         const AccessorRW<double,2> fa_ghostx,
                                         const AccessorRW<double,2> fa_ghosty,
                                         const AccessorRW<double,2> fa_ghostz,
                                         const AccessorRO<double,3> fa_qim,
                                         const int x_range, const int y_range, 
                                         const int z_range, const int corner,
                                         const bool stride_x_positive,
                                         const bool stride_y_positive,
                                         const bool stride_z_positive,
                                         const bool mms_source, 
                                         const int num_moments, 
                                         const int num_wavefronts,
                                         const double hi, const double hj,
                                         const double hk, const double vdelt,
                                         const double *device_ec,
                                         const double *device_mu,
                                         const double *device_eta,
                                         const double *device_xi,
                                         const double *device_w,
                                         volatile double *flux_x,
                                         volatile double *flux_y,
                                         volatile double *flux_z,
                                         const int *wave_length,
                                         const int *wave_offset,
                                         const int *wave_x,
                                         const int *wave_y,
                                         const int *wave_z,
                                         volatile int *mutexin,
                                         volatile int *mutexout)
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
  unsigned warpid = threadIdx.x >> 5;

  const double tolr = 1.0e-12;

  int barval = gridDim.x;

  for (int w = 0; w < num_wavefronts; w++) {
    const int wave_size = wave_length[w];
    const int wave_off = wave_offset[w];
    for (int n = blockIdx.x; n < wave_size; n += gridDim.x) {
      // Figure out the local point that we are working on    
      Point<3> local_point = origin;
      const int x = wave_x[wave_off+n];
      const int y = wave_y[wave_off+n];
      const int z = wave_z[wave_off+n];
      local_point[0] += x;
      local_point[1] += y;
      local_point[2] += z;

      // Compute the angular source
      MomentQuad quad = fa_qtot[local_point];
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
          psi[ang] += angle_read<3>(fa_qim, local_point, ang);
      }

      // Compute the initial solution
      #pragma unroll
      for (int ang = 0; ang < THR_ANGLES; ang++)
        pc[ang] = psi[ang];
      // X ghost cells
      if ((stride_x_positive && (x == 0)) ||
          (!stride_x_positive && (x == (x_range-1)))) {
        // Ghost cell array
        Point<2> ghost_point = ghostx_point(local_point); 
        #pragma unroll 
        for (int ang = 0; ang < THR_ANGLES; ang++)
          psii[ang] = angle_read<2>(fa_ghostx, ghost_point, ang);
      } else {
        // Local array
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          psii[ang] = read_flux(flux_x, y, z, z_range, num_angles, ang);
      }
      #pragma unroll
      for (int ang = 0; ang < THR_ANGLES; ang++)
        pc[ang] += psii[ang] * device_mu[ang*blockDim.x + threadIdx.x] * hi;
      // Y ghost cells
      if ((stride_y_positive && (y == 0)) || 
          (!stride_y_positive && (y == (y_range-1)))) {
        // Ghost cell array
        Point<2> ghost_point = ghosty_point(local_point);
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          psij[ang] = angle_read<2>(fa_ghosty, ghost_point, ang);
      } else {
        // Local array
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          psij[ang] = read_flux(flux_y, x, z, z_range, num_angles, ang);
      }
      #pragma unroll
      for (int ang = 0; ang < THR_ANGLES; ang++)
        pc[ang] += psij[ang] * device_eta[ang * blockDim.x + threadIdx.x] * hj;
      // Z ghost cells
      if ((stride_z_positive && (z == 0)) ||
          (!stride_z_positive && (z == (z_range-1)))) {
        // Ghost cell array
        Point<2> ghost_point = ghostz_point(local_point);
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          psik[ang] = angle_read<2>(fa_ghostz, ghost_point, ang);
      } else {
        // Local array
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          psik[ang] = read_flux(flux_z, x, y, y_range, num_angles, ang);
      }
      #pragma unroll
      for (int ang = 0; ang < THR_ANGLES; ang++)
        pc[ang] += psik[ang] * device_xi[ang * blockDim.x + threadIdx.x] * hk;

      #pragma unroll
      for (int ang = 0; ang < THR_ANGLES; ang++) {
        time_flux_in[ang] = angle_read<3>(fa_time_flux_in, local_point, ang);
        pc[ang] += vdelt * time_flux_in[ang];
      }
      
      #pragma unroll
      for (int ang = 0; ang < THR_ANGLES; ang++) {
        double dinv = angle_read<3>(fa_dinv, local_point, ang);
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

      const double t_xs = fa_t_xs[local_point];
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
          negative_fluxes += __shfl_xor_sync(0xffffffff, negative_fluxes, i, 32);
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
          negative_fluxes += __shfl_xor_sync(0xffffffff, negative_fluxes, i, 32);
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
        angle_write<3>(fa_time_flux_out, local_point, ang, time_flux_out);
      }
      // Write out the ghost regions
      // X ghost
      if ((stride_x_positive && (x == (x_range - 1))) ||
          (!stride_x_positive && (x == 0))) {
        Point<2> ghost_point = ghostx_point(local_point);
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          angle_write<2>(fa_ghostx, ghost_point, ang, psii[ang]);
      } else {
        // Write to local flux
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          write_flux(flux_x, y, z, z_range, num_angles, ang, psii[ang]);
      }
      // Y ghost
      if ((stride_y_positive && (y == (y_range - 1))) ||
          (!stride_y_positive && (y == 0))) {
        Point<2> ghost_point = ghosty_point(local_point);
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          angle_write<2>(fa_ghosty, ghost_point, ang, psij[ang]);
      } else {
        // Write to the pencil
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          write_flux(flux_y, x, z, z_range, num_angles, ang, psij[ang]);
      }
      // Z ghost
      if ((stride_z_positive && (z == (z_range - 1))) ||
          (!stride_z_positive && (z == 0))) {
        Point<2> ghost_point = ghostz_point(local_point);
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          angle_write<2>(fa_ghostz, ghost_point, ang, psik[ang]);
      } else {
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          write_flux(flux_z, x, y, y_range, num_angles, ang, psik[ang]);
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
      for (int i = 16; i >= 1; i /= 2)
        total += __shfl_xor_sync(0xffffffff, total, i, 32);
      if (warpid == 0)
        double_trampoline[laneid] = 0.0;
      __syncthreads();
      if (laneid == 0)
        double_trampoline[warpid] = total;
      __syncthreads();
      if (warpid == 0) {
        total = double_trampoline[laneid];
        #pragma unroll
        for (int i = 16; i >= 1; i /= 2)
          total += __shfl_xor_sync(0xffffffff, total, i, 32);
        // Do the reduction
        if (laneid == 0) {
          double *local_flux = fa_flux.ptr(local_point); 
          ourAtomicAdd(local_flux, total);
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
          for (int i = 16; i >= 1; i /= 2)
            total += __shfl_xor_sync(0xffffffff, total, i, 32);
          if (laneid == 0) {
            double *local_fluxm = (double*)fa_fluxm.ptr(local_point);
            local_fluxm += (l-1);
            ourAtomicAdd(local_fluxm, total);
          }
        }
      }
    }
    if (gridDim.x > 1) {
      kernel_barrier(barval, mutexin, mutexout);
      barval += gridDim.x;
    } else {
      __syncthreads();
    }
  }
}

template<int THR_ANGLES>
__global__
void gpu_time_dependent_sweep_without_fixup(const Point<3> origin, 
                                         const AccessorRO<MomentQuad,3> fa_qtot,
                                         const AccessorRW<double,3> fa_flux,
                                         const AccessorRW<MomentTriple,3> fa_fluxm,
                                         const AccessorRO<double,3> fa_dinv,
                                         const AccessorRO<double,3> fa_time_flux_in,
                                         const AccessorWO<double,3> fa_time_flux_out,
                                         const AccessorRO<double,3> fa_t_xs,
                                         const AccessorRW<double,2> fa_ghostx,
                                         const AccessorRW<double,2> fa_ghosty,
                                         const AccessorRW<double,2> fa_ghostz,
                                         const AccessorRO<double,3> fa_qim,
                                         const int x_range, const int y_range, 
                                         const int z_range, const int corner,
                                         const bool stride_x_positive,
                                         const bool stride_y_positive,
                                         const bool stride_z_positive,
                                         const bool mms_source, 
                                         const int num_moments, 
                                         const int num_wavefronts,
                                         const double hi, const double hj,
                                         const double hk, const double vdelt,
                                         const double *device_ec,
                                         const double *device_mu,
                                         const double *device_eta,
                                         const double *device_xi,
                                         const double *device_w,
                                         volatile double *flux_x,
                                         volatile double *flux_y,
                                         volatile double *flux_z,
                                         const int *wave_length,
                                         const int *wave_offset,
                                         const int *wave_x,
                                         const int *wave_y,
                                         const int *wave_z,
                                         volatile int *mutexin,
                                         volatile int *mutexout)
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
  unsigned warpid = threadIdx.x >> 5;

  int barval = gridDim.x;

  for (int w = 0; w < num_wavefronts; w++) {
    const int wave_size = wave_length[w];
    const int wave_off = wave_offset[w];
    for (int n = blockIdx.x; n < wave_size; n += gridDim.x) {
      // Figure out the local point that we are working on    
      Point<3> local_point = origin;
      const int x = wave_x[wave_off+n];
      const int y = wave_y[wave_off+n];
      const int z = wave_z[wave_off+n];
      local_point[0] += x;
      local_point[1] += y;
      local_point[2] += z;

      // Compute the angular source
      MomentQuad quad = fa_qtot[local_point];
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
          psi[ang] += angle_read<3>(fa_qim, local_point, ang);
      }

      // Compute the initial solution
      #pragma unroll
      for (int ang = 0; ang < THR_ANGLES; ang++)
        pc[ang] = psi[ang];
      // X ghost cells
      if ((stride_x_positive && (x == 0)) ||
          (!stride_x_positive && (x == (x_range-1)))) {
        // Ghost cell array
        Point<2> ghost_point = ghostx_point(local_point);
        #pragma unroll 
        for (int ang = 0; ang < THR_ANGLES; ang++)
          psii[ang] = angle_read<2>(fa_ghostx, ghost_point, ang);
      } else {
        // Local array  
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          psii[ang] = read_flux(flux_x, y, z, z_range, num_angles, ang);
      }
      #pragma unroll
      for (int ang = 0; ang < THR_ANGLES; ang++)
        pc[ang] += psii[ang] * device_mu[ang*blockDim.x + threadIdx.x] * hi;
      // Y ghost cells
      if ((stride_y_positive && (y == 0)) ||
          (!stride_y_positive && (y == (y_range-1)))) {
        // Ghost cell array
        Point<2> ghost_point = ghosty_point(local_point);
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          psij[ang] = angle_read<2>(fa_ghosty, ghost_point, ang);
      } else {
        // Local array
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          psij[ang] = read_flux(flux_y, x, z, z_range, num_angles, ang);
      }
      #pragma unroll
      for (int ang = 0; ang < THR_ANGLES; ang++)
        pc[ang] += psij[ang] * device_eta[ang * blockDim.x + threadIdx.x] * hj;
      // Z ghost cells
      if ((stride_z_positive && (z == 0)) ||
          (!stride_z_positive && (z == (z_range-1)))) {
        // Ghost cell array
        Point<2> ghost_point = ghostz_point(local_point);
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          psik[ang] = angle_read<2>(fa_ghostz, ghost_point, ang);
      } else {
        // Local array
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          psik[ang] = read_flux(flux_z, x, y, y_range, num_angles, ang);
      }
      #pragma unroll
      for (int ang = 0; ang < THR_ANGLES; ang++)
        pc[ang] += psik[ang] * device_xi[ang * blockDim.x + threadIdx.x] * hk;

      #pragma unroll
      for (int ang = 0; ang < THR_ANGLES; ang++) {
        time_flux_in[ang] = angle_read<3>(fa_time_flux_in, local_point, ang);
        pc[ang] += vdelt * time_flux_in[ang];
      }
      
      #pragma unroll
      for (int ang = 0; ang < THR_ANGLES; ang++) {
        double dinv = angle_read<3>(fa_dinv, local_point, ang);
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
        angle_write<3>(fa_time_flux_out, local_point, ang, time_flux_out);
      }
      // Write out the ghost regions
      // X ghost
      if ((stride_x_positive && (x == (x_range - 1))) ||
          (!stride_x_positive && (x == 0))) {
        Point<2> ghost_point = ghostx_point(local_point);
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          angle_write<2>(fa_ghostx, ghost_point, ang, psii[ang]);
      } else {
        // Write to local flux
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          write_flux(flux_x, y, z, z_range, num_angles, ang, psii[ang]);
      }
      // Y ghost
      if ((stride_y_positive && (y == (y_range - 1))) ||
          (!stride_y_positive && (y == 0))) {
        Point<2> ghost_point = ghosty_point(local_point);
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          angle_write<2>(fa_ghosty, ghost_point, ang, psij[ang]);
      } else {
        // Write to the pencil
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          write_flux(flux_y, x, z, z_range, num_angles, ang, psij[ang]);
      }
      // Z ghost
      if ((stride_z_positive && (z == (z_range - 1))) ||
          (!stride_z_positive && (z == 0))) {
        Point<2> ghost_point = ghostz_point(local_point);
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          angle_write<2>(fa_ghostz, ghost_point, ang, psik[ang]);
      } else {
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          write_flux(flux_z, x, y, y_range, num_angles, ang, psik[ang]);
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
      for (int i = 16; i >= 1; i /= 2)
        total += __shfl_xor_sync(0xffffffff, total, i, 32);
      if (warpid == 0)
        double_trampoline[laneid] = 0.0;
      __syncthreads();
      if (laneid == 0)
        double_trampoline[warpid] = total;
      __syncthreads();
      if (warpid == 0) {
        total = double_trampoline[laneid];
        #pragma unroll
        for (int i = 16; i >= 1; i /= 2)
          total += __shfl_xor_sync(0xffffffff, total, i, 32);
        // Do the reduction
        if (laneid == 0) {
          double *local_flux = fa_flux.ptr(local_point);
          ourAtomicAdd(local_flux, total);
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
          for (int i = 16; i >= 1; i /= 2)
            total += __shfl_xor_sync(0xffffffff, total, i, 32);
          if (laneid == 0) {
            double *local_fluxm = (double*)fa_fluxm.ptr(local_point);
            local_fluxm += (l-1);
            ourAtomicAdd(local_fluxm, total);
          }
        }
      }
    }
    if (gridDim.x > 1) {
      kernel_barrier(barval, mutexin, mutexout);
      barval += gridDim.x;
    } else {
      __syncthreads();
    }
  }
}

template<int THR_ANGLES>
__global__
void gpu_time_independent_sweep_with_fixup(const Point<3> origin, 
                                         const AccessorRO<MomentQuad,3> fa_qtot,
                                         const AccessorRW<double,3> fa_flux,
                                         const AccessorRW<MomentTriple,3> fa_fluxm,
                                         const AccessorRO<double,3> fa_dinv,
                                         const AccessorRO<double,3> fa_t_xs,
                                         const AccessorRW<double,2> fa_ghostx,
                                         const AccessorRW<double,2> fa_ghosty,
                                         const AccessorRW<double,2> fa_ghostz,
                                         const AccessorRO<double,3> fa_qim,
                                         const int x_range, const int y_range, 
                                         const int z_range, const int corner,
                                         const bool stride_x_positive,
                                         const bool stride_y_positive,
                                         const bool stride_z_positive,
                                         const bool mms_source, 
                                         const int num_moments, 
                                         const int num_wavefronts,
                                         const double hi, const double hj,
                                         const double hk,
                                         const double *device_ec,
                                         const double *device_mu,
                                         const double *device_eta,
                                         const double *device_xi,
                                         const double *device_w,
                                         volatile double *flux_x,
                                         volatile double *flux_y,
                                         volatile double *flux_z,
                                         const int *wave_length,
                                         const int *wave_offset,
                                         const int *wave_x,
                                         const int *wave_y,
                                         const int *wave_z,
                                         volatile int *mutexin,
                                         volatile int *mutexout)
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
  unsigned warpid = threadIdx.x >> 5;

  const double tolr = 1.0e-12;

  int barval = gridDim.x;

  for (int w = 0; w < num_wavefronts; w++) {
    const int wave_size = wave_length[w];
    const int wave_off = wave_offset[w];
    for (int n = blockIdx.x; n < wave_size; n += gridDim.x) {
      // Figure out the local point that we are working on    
      Point<3> local_point = origin;
      const int x = wave_x[wave_off+n];
      const int y = wave_y[wave_off+n];
      const int z = wave_z[wave_off+n];
      local_point[0] += x;
      local_point[1] += y;
      local_point[2] += z;

      // Compute the angular source
      MomentQuad quad = fa_qtot[local_point];
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
          psi[ang] += angle_read<3>(fa_qim, local_point, ang);
      }

      // Compute the initial solution
      #pragma unroll
      for (int ang = 0; ang < THR_ANGLES; ang++)
        pc[ang] = psi[ang];
      // X ghost cells
      if ((stride_x_positive && (x == 0)) ||
          (!stride_x_positive && (x == (x_range-1)))) {
        // Ghost cell array
        Point<2> ghost_point = ghostx_point(local_point); 
        #pragma unroll 
        for (int ang = 0; ang < THR_ANGLES; ang++)
          psii[ang] = angle_read<2>(fa_ghostx, ghost_point, ang);
      } else {
        // Local array
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          psii[ang] = read_flux(flux_x, y, z, z_range, num_angles, ang);
      }
      #pragma unroll
      for (int ang = 0; ang < THR_ANGLES; ang++)
        pc[ang] += psii[ang] * device_mu[ang*blockDim.x + threadIdx.x] * hi;
      // Y ghost cells
      if ((stride_y_positive && (y == 0)) ||
          (!stride_y_positive && (y == (y_range-1)))) {
        // Ghost cell array
        Point<2> ghost_point = ghosty_point(local_point);
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          psij[ang] = angle_read<2>(fa_ghosty, ghost_point, ang);
      } else {
        // Local array
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          psij[ang] = read_flux(flux_y, x, z, z_range, num_angles, ang);
      }
      #pragma unroll
      for (int ang = 0; ang < THR_ANGLES; ang++)
        pc[ang] += psij[ang] * device_eta[ang * blockDim.x + threadIdx.x] * hj;
      // Z ghost cells
      if ((stride_z_positive && (z == 0)) ||
          (!stride_z_positive && (z == (z_range-1)))) {
        // Ghost cell array
        Point<2> ghost_point = ghostz_point(local_point);
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          psik[ang] = angle_read<2>(fa_ghostz, ghost_point, ang);
      } else {
        // Local array
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          psik[ang] = read_flux(flux_z, x, y, y_range, num_angles, ang);
      }
      #pragma unroll
      for (int ang = 0; ang < THR_ANGLES; ang++)
        pc[ang] += psik[ang] * device_xi[ang * blockDim.x + threadIdx.x] * hk;

      #pragma unroll
      for (int ang = 0; ang < THR_ANGLES; ang++) {
        double dinv = angle_read<3>(fa_dinv, local_point, ang);
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

      const double t_xs = fa_t_xs[local_point];
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
          negative_fluxes += __shfl_xor_sync(0xffffffff, negative_fluxes, i, 32);
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
          negative_fluxes += __shfl_xor_sync(0xffffffff, negative_fluxes, i, 32);
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
      if ((stride_x_positive && (x == (x_range - 1))) ||
          (!stride_x_positive && (x == 0))) {
        Point<2> ghost_point = ghostx_point(local_point);
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          angle_write<2>(fa_ghostx, ghost_point, ang, psii[ang]);
      } else {
        // Write to local flux
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          write_flux(flux_x, y, z, z_range, num_angles, ang, psii[ang]);
      }
      // Y ghost
      if ((stride_y_positive && (y == (y_range - 1))) ||
          (!stride_y_positive && (y == 0))) {
        Point<2> ghost_point = ghosty_point(local_point);
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          angle_write<2>(fa_ghosty, ghost_point, ang, psij[ang]);
      } else {
        // Write to the pencil
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          write_flux(flux_y, x, z, z_range, num_angles, ang, psij[ang]);
      }
      // Z ghost
      if ((stride_z_positive && (z == (z_range - 1))) ||
          (!stride_z_positive && (z == 0))) {
        Point<2> ghost_point = ghostz_point(local_point);
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          angle_write<2>(fa_ghostz, ghost_point, ang, psik[ang]);
      } else {
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          write_flux(flux_z, x, y, y_range, num_angles, ang, psik[ang]);
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
      for (int i = 16; i >= 1; i /= 2)
        total += __shfl_xor_sync(0xffffffff, total, i, 32);
      if (warpid == 0)
        double_trampoline[laneid] = 0.0;
      __syncthreads();
      if (laneid == 0)
        double_trampoline[warpid] = total;
      __syncthreads();
      if (warpid == 0) {
        total = double_trampoline[laneid];
        #pragma unroll
        for (int i = 16; i >= 1; i /= 2)
          total += __shfl_xor_sync(0xffffffff, total, i, 32);
        // Do the reduction
        if (laneid == 0) {
          double *local_flux = fa_flux.ptr(local_point);
          ourAtomicAdd(local_flux, total);
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
          for (int i = 16; i >= 1; i /= 2)
            total += __shfl_xor_sync(0xffffffff, total, i, 32);
          if (laneid == 0) {
            double *local_fluxm = (double*)fa_fluxm.ptr(local_point);
            local_fluxm += (l-1);
            ourAtomicAdd(local_fluxm, total);
          }
        }
      }
    }
    if (gridDim.x > 1) {
      kernel_barrier(barval, mutexin, mutexout);
      barval += gridDim.x;
    } else {
      __syncthreads();
    }
  }
}

template<int THR_ANGLES>
__global__
void gpu_time_independent_sweep_without_fixup(const Point<3> origin, 
                                         const AccessorRO<MomentQuad,3> fa_qtot,
                                         const AccessorRW<double,3> fa_flux,
                                         const AccessorRW<MomentTriple,3> fa_fluxm,
                                         const AccessorRO<double,3> fa_dinv,
                                         const AccessorRO<double,3> fa_t_xs,
                                         const AccessorRW<double,2> fa_ghostx,
                                         const AccessorRW<double,2> fa_ghosty,
                                         const AccessorRW<double,2> fa_ghostz,
                                         const AccessorRO<double,3> fa_qim,
                                         const int x_range, const int y_range, 
                                         const int z_range, const int corner,
                                         const bool stride_x_positive,
                                         const bool stride_y_positive,
                                         const bool stride_z_positive,
                                         const bool mms_source, 
                                         const int num_moments, 
                                         const int num_wavefronts,
                                         const double hi, const double hj,
                                         const double hk,
                                         const double *device_ec,
                                         const double *device_mu,
                                         const double *device_eta,
                                         const double *device_xi,
                                         const double *device_w,
                                         volatile double *flux_x,
                                         volatile double *flux_y,
                                         volatile double *flux_z,
                                         const int *wave_length,
                                         const int *wave_offset,
                                         const int *wave_x,
                                         const int *wave_y,
                                         const int *wave_z,
                                         volatile int *mutexin,
                                         volatile int *mutexout) 
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
  unsigned warpid = threadIdx.x >> 5;

  int barval = gridDim.x;

  for (int w = 0; w < num_wavefronts; w++) {
    const int wave_size = wave_length[w];
    const int wave_off = wave_offset[w];
    for (int n = blockIdx.x; n < wave_size; n += gridDim.x) {
      // Figure out the local point that we are working on    
      Point<3> local_point = origin;
      const int x = wave_x[wave_off+n];
      const int y = wave_y[wave_off+n];
      const int z = wave_z[wave_off+n];
      local_point[0] += x;
      local_point[1] += y;
      local_point[2] += z;

      // Compute the angular source
      MomentQuad quad = fa_qtot[local_point];
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
          psi[ang] += angle_read<3>(fa_qim, local_point, ang);
      }

      // Compute the initial solution
      #pragma unroll
      for (int ang = 0; ang < THR_ANGLES; ang++)
        pc[ang] = psi[ang];
      // X ghost cells
      if ((stride_x_positive && (x == 0)) ||
          (!stride_x_positive && (x == (x_range-1)))) {
        // Ghost cell array
        Point<2> ghost_point = ghostx_point(local_point); 
        #pragma unroll 
        for (int ang = 0; ang < THR_ANGLES; ang++)
          psii[ang] = angle_read<2>(fa_ghostx, ghost_point, ang);
      } else {
        // Local array
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          psii[ang] = read_flux(flux_x, y, z, z_range, num_angles, ang);
      }
      #pragma unroll
      for (int ang = 0; ang < THR_ANGLES; ang++)
        pc[ang] += psii[ang] * device_mu[ang*blockDim.x + threadIdx.x] * hi;
      // Y ghost cells
      if ((stride_y_positive && (y == 0)) ||
          (!stride_y_positive && (y == (y_range-1)))) {
        // Ghost cell array
        Point<2> ghost_point = ghosty_point(local_point);
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          psij[ang] = angle_read<2>(fa_ghosty, ghost_point, ang);
      } else {
        // Local array
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          psij[ang] = read_flux(flux_y, x, z, z_range, num_angles, ang);
      }
      #pragma unroll
      for (int ang = 0; ang < THR_ANGLES; ang++)
        pc[ang] += psij[ang] * device_eta[ang * blockDim.x + threadIdx.x] * hj;
      // Z ghost cells
      if ((stride_z_positive && (z == 0)) ||
          (!stride_z_positive && (z == (z_range-1)))) {
        // Ghost cell array
        Point<2> ghost_point = ghostz_point(local_point);
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          psik[ang] = angle_read<2>(fa_ghostz, ghost_point, ang);
      } else {
        // Local array
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          psik[ang] = read_flux(flux_z, x, y, y_range, num_angles, ang);
      }
      #pragma unroll
      for (int ang = 0; ang < THR_ANGLES; ang++)
        pc[ang] += psik[ang] * device_xi[ang * blockDim.x + threadIdx.x] * hk;

      #pragma unroll
      for (int ang = 0; ang < THR_ANGLES; ang++) {
        double dinv = angle_read<3>(fa_dinv, local_point, ang);
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
      if ((stride_x_positive && (x == (x_range - 1))) ||
          (!stride_x_positive && (x == 0))) {
        Point<2> ghost_point = ghostx_point(local_point);
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          angle_write<2>(fa_ghostx, ghost_point, ang, psii[ang]);
      } else {
        // Write to local flux
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          write_flux(flux_x, y, z, z_range, num_angles, ang, psii[ang]);
      }
      // Y ghost
      if ((stride_y_positive && (y == (y_range - 1))) ||
          (!stride_y_positive && (y == 0))) {
        Point<2> ghost_point = ghosty_point(local_point);
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          angle_write<2>(fa_ghosty, ghost_point, ang, psij[ang]);
      } else {
        // Write to the pencil
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          write_flux(flux_y, x, z, z_range, num_angles, ang, psij[ang]);
      }
      // Z ghost
      if ((stride_z_positive && (z == (z_range - 1))) ||
          (!stride_z_positive && (z == 0))) {
        Point<2> ghost_point = ghostz_point(local_point);
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          angle_write<2>(fa_ghostz, ghost_point, ang, psik[ang]);
      } else {
        #pragma unroll
        for (int ang = 0; ang < THR_ANGLES; ang++)
          write_flux(flux_z, x, y, y_range, num_angles, ang, psik[ang]);
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
      for (int i = 16; i >= 1; i /= 2)
        total += __shfl_xor_sync(0xffffffff, total, i, 32);
      if (warpid == 0)
        double_trampoline[laneid] = 0.0;
      __syncthreads();
      if (laneid == 0)
        double_trampoline[warpid] = total;
      __syncthreads();
      if (warpid == 0) {
        total = double_trampoline[laneid];
        #pragma unroll
        for (int i = 16; i >= 1; i /= 2)
          total += __shfl_xor_sync(0xffffffff, total, i, 32);
        // Do the reduction
        if (laneid == 0) {
          double *local_flux = fa_flux.ptr(local_point);
          ourAtomicAdd(local_flux, total);
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
          for (int i = 16; i >= 1; i /= 2)
            total += __shfl_xor_sync(0xffffffff, total, i, 32);
          if (laneid == 0) {
            double *local_fluxm = (double*)fa_fluxm.ptr(local_point);
            local_fluxm += (l-1);
            ourAtomicAdd(local_fluxm, total);
          }
        }
      }
    }
    if (gridDim.x > 1) {
      kernel_barrier(barval, mutexin, mutexout);
      barval += gridDim.x;
    } else {
      __syncthreads();
    }
  }
}

__host__
int compute_blocks_per_sweep(int nx, int ny, int nz,
                             const void *func, int threads_per_block,
                             Runtime *runtime, Context ctx, int gpu) 
{
  int numsm;
  cudaDeviceGetAttribute(&numsm, cudaDevAttrMultiProcessorCount, gpu);
  Future batches = runtime->select_tunable_value(ctx, Snap::GPU_SMS_PER_SWEEP_TUNABLE, 
      0/*mapper id*/, 0/*mapper tag*/, &numsm, sizeof(numsm));
  int active_blocks_per_sm;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks_per_sm,
      func, threads_per_block, 0);
  int numsm_per_kernel = batches.get_result<int>(true/*silence warnings*/); 
  int grid_size = active_blocks_per_sm * numsm_per_kernel;
  // Clamp this to the max wavefront length
  if (grid_size > max_wavefront_length[gpu])
    grid_size = max_wavefront_length[gpu];
  for (int idx = 0; idx < MAX_STREAMS; idx++)
    if (cudaMalloc((void**)&mutex_in_d[gpu][idx], grid_size*sizeof(int)) 
        != cudaSuccess)
    {
      printf("ERROR: failed to allocate mutex in data structure of %zd bytes on gpu %d\n",
          grid_size * sizeof(int), gpu);
      exit(1);
    }
  for (int idx = 0; idx < MAX_STREAMS; idx++)
    if (cudaMalloc((void**)&mutex_out_d[gpu][idx], grid_size*sizeof(int)) 
        != cudaSuccess)
    {
      printf("ERROR: failed to allocate mutex out data structure of %zd bytes on gpu %d\n",
          grid_size * sizeof(int), gpu);
      exit(1);
    }
  return grid_size; 
}

__host__
void run_gpu_sweep(const Point<3> origin, 
               const AccessorRO<MomentQuad,3> &fa_qtot,
               const AccessorRW<double,3> &fa_flux,
               const AccessorRW<MomentTriple,3> &fa_fluxm,
               const AccessorRO<double,3> &fa_dinv,
               const AccessorRO<double,3> &fa_time_flux_in,
               const AccessorWO<double,3> &fa_time_flux_out,
               const AccessorRO<double,3> &fa_t_xs,
               const AccessorRW<double,2> &fa_ghostx,
               const AccessorRW<double,2> &fa_ghosty,
               const AccessorRW<double,2> &fa_ghostz,
               const AccessorRO<double,3> &fa_qim,
               const int x_range, const int y_range, 
               const int z_range, const int corner,
               const bool stride_x_positive,
               const bool stride_y_positive,
               const bool stride_z_positive,
               const bool mms_source, 
               const int num_moments, 
               const double hi, const double hj,
               const double hk, const double vdelt,
               const int num_angles, const bool fixup, 
               Runtime *runtime, Context ctx)
{
  // Figure out how many angles per thread we need
  const int max_threads_per_cta = 1024;
  const int angles_per_thread = 
    (num_angles + max_threads_per_cta - 1) / max_threads_per_cta;
  // Have to be evenly divisible for now
  assert((num_angles % angles_per_thread) == 0);
  const int threads_per_block = num_angles / angles_per_thread;
  dim3 block(threads_per_block, 1, 1);
  int gpu;
  cudaGetDevice(&gpu);
  dim3 grid(blocks_per_sweep[gpu], 1, 1);
  // Put each sweep on its own stream since they are independent
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  int stream_idx = -1;
  for (int idx = 0; idx < MAX_STREAMS; idx++)
  {
    if (flux_streams[gpu][idx] == stream)
    {
      stream_idx = idx;
      break;
    }
    else if (flux_streams[gpu][idx] == 0)
    {
      flux_streams[gpu][idx] = stream;
      stream_idx = idx;
      break;
    }
  }
  assert(stream_idx >= 0);
  // No need to delete the stream, Realm CUDA hijack takes care of it
  if (fixup) {
    // Need fixup
    if (vdelt != 0.0) {
      // Time dependent
      const void *args[] = { &origin,
                &fa_qtot, &fa_flux, &fa_fluxm, &fa_dinv, &fa_time_flux_in,
                &fa_time_flux_out, &fa_t_xs, &fa_ghostx, &fa_ghosty,
                &fa_ghostz, &fa_qim,
                &x_range, &y_range, &z_range, &corner, &stride_x_positive, 
                &stride_y_positive, &stride_z_positive, &mms_source, 
                &num_moments, &total_wavefronts[gpu], &hi, &hj, &hk, &vdelt,
                &ec_d[gpu], &mu_d[gpu], &eta_d[gpu], &xi_d[gpu], &w_d[gpu],
                &flux_x_d[gpu][stream_idx], &flux_y_d[gpu][stream_idx],
                &flux_z_d[gpu][stream_idx], &wavefront_length_d[gpu][corner],
                &wavefront_offset_d[gpu][corner], &wavefront_x_d[gpu][corner],
                &wavefront_y_d[gpu][corner], &wavefront_z_d[gpu][corner],
                &mutex_in_d[gpu][stream_idx], &mutex_out_d[gpu][stream_idx] };
      switch (angles_per_thread)
      {
        case 1:
          {
            if (grid.x == 0)
            {
              grid.x = compute_blocks_per_sweep(x_range, y_range, z_range,            
                  (const void*)gpu_time_dependent_sweep_with_fixup<1>, 
                  threads_per_block, runtime, ctx, gpu);
              blocks_per_sweep[gpu] = grid.x;
            }
            cudaLaunchCooperativeKernel((const void*)gpu_time_dependent_sweep_with_fixup<1>,
                grid, block, (void**)args, 0, stream);
            break;
          }
        case 2:
          {
            if (grid.x == 0)
            {
              grid.x = compute_blocks_per_sweep(x_range, y_range, z_range,            
                  (const void*)gpu_time_dependent_sweep_with_fixup<2>, 
                  threads_per_block, runtime, ctx, gpu);
              blocks_per_sweep[gpu] = grid.x;
            }
            cudaLaunchCooperativeKernel((const void*)gpu_time_dependent_sweep_with_fixup<2>,
                grid, block, (void**)args, 0, stream);
            break;
          }
        default:
          printf("OH SNAP! That is a lot of angles! Add more cases!\n");
          exit(1);
      }
    } else {
      // Time independent
      const void *args[] = { &origin,
                &fa_qtot, &fa_flux, &fa_fluxm, &fa_dinv, &fa_t_xs,
                &fa_ghostx, &fa_ghosty, &fa_ghostz, &fa_qim,
                &x_range, &y_range, &z_range, &corner, 
                &stride_x_positive, &stride_y_positive, &stride_z_positive,
                &mms_source, &num_moments, &total_wavefronts[gpu], &hi, &hj, &hk,
                &ec_d[gpu], &mu_d[gpu], &eta_d[gpu], &xi_d[gpu], &w_d[gpu],
                &flux_x_d[gpu][stream_idx], &flux_y_d[gpu][stream_idx],
                &flux_z_d[gpu][stream_idx], &wavefront_length_d[gpu][corner],
                &wavefront_offset_d[gpu][corner], &wavefront_x_d[gpu][corner],
                &wavefront_y_d[gpu][corner], &wavefront_z_d[gpu][corner],
                &mutex_in_d[gpu][stream_idx], &mutex_out_d[gpu][stream_idx] };
      switch (angles_per_thread)
      {
        case 1:
          {
            if (grid.x == 0)
            {
              grid.x = compute_blocks_per_sweep(x_range, y_range, z_range,            
                  (const void*)gpu_time_independent_sweep_with_fixup<1>, 
                  threads_per_block, runtime, ctx, gpu);
              blocks_per_sweep[gpu] = grid.x;
            }
            cudaLaunchCooperativeKernel((const void*)gpu_time_independent_sweep_with_fixup<1>,
                grid, block, (void**)args, 0, stream);
            break;
          }
        case 2:
          {
            if (grid.x == 0)
            {
              grid.x = compute_blocks_per_sweep(x_range, y_range, z_range,            
                  (const void*)gpu_time_independent_sweep_with_fixup<2>, 
                  threads_per_block, runtime, ctx, gpu);
              blocks_per_sweep[gpu] = grid.x;
            }
            cudaLaunchCooperativeKernel((const void*)gpu_time_independent_sweep_with_fixup<2>,
                grid, block, (void**)args, 0, stream);
            break;
          }
        default:
          printf("ON SNAP! That is a lot of angles! Add more cases!\n");
          exit(1);
      }
    }
  } else {
    // No fixup
    if (vdelt != 0.0) {
      // Time dependent
      const void *args[] = { &origin, 
                &fa_qtot, &fa_flux, &fa_fluxm, &fa_dinv, &fa_time_flux_in,
                &fa_time_flux_out, &fa_t_xs, &fa_ghostx, &fa_ghosty, &fa_ghostz, &fa_qim,
                &x_range, &y_range, &z_range, &corner, 
                &stride_x_positive, &stride_y_positive, &stride_z_positive, 
                &mms_source, &num_moments, &total_wavefronts[gpu], &hi, &hj, &hk, &vdelt,
                &ec_d[gpu], &mu_d[gpu], &eta_d[gpu], &xi_d[gpu], &w_d[gpu],
                &flux_x_d[gpu][stream_idx], &flux_y_d[gpu][stream_idx],
                &flux_z_d[gpu][stream_idx], &wavefront_length_d[gpu][corner],
                &wavefront_offset_d[gpu][corner], &wavefront_x_d[gpu][corner],
                &wavefront_y_d[gpu][corner], &wavefront_z_d[gpu][corner],
                &mutex_in_d[gpu][stream_idx], &mutex_out_d[gpu][stream_idx] };
      switch (angles_per_thread)
      {
        case 1:
          {
            if (grid.x == 0)
            {
              grid.x = compute_blocks_per_sweep(x_range, y_range, z_range,            
                  (const void*)gpu_time_dependent_sweep_without_fixup<1>, 
                  threads_per_block, runtime, ctx, gpu);
              blocks_per_sweep[gpu] = grid.x;
            }
            cudaLaunchCooperativeKernel((const void*)gpu_time_dependent_sweep_without_fixup<1>,
                grid, block, (void**)args, 0, stream);
            break;
          }
        case 2:
          {
            if (grid.x == 0)
            {
              grid.x = compute_blocks_per_sweep(x_range, y_range, z_range,            
                  (const void*)gpu_time_dependent_sweep_without_fixup<2>, 
                  threads_per_block, runtime, ctx, gpu);
              blocks_per_sweep[gpu] = grid.x;
            }
            cudaLaunchCooperativeKernel((const void*)gpu_time_dependent_sweep_without_fixup<2>,
                grid, block, (void**)args, 0, stream);
            break;
          }
        default:
          printf("OH SNAP! That is a lot of angles! Add more cases!\n");
          exit(1);
      }
    } else {
      // Time independent
      const void *args[] = { &origin,
                &fa_qtot, &fa_flux, &fa_fluxm, &fa_dinv, &fa_t_xs,
                &fa_ghostx, &fa_ghosty, &fa_ghostz, &fa_qim,
                &x_range, &y_range, &z_range, &corner, 
                &stride_x_positive, &stride_y_positive, &stride_z_positive,
                &mms_source, &num_moments, &total_wavefronts[gpu], &hi, &hj, &hk,
                &ec_d[gpu], &mu_d[gpu], &eta_d[gpu], &xi_d[gpu], &w_d[gpu],
                &flux_x_d[gpu][stream_idx], &flux_y_d[gpu][stream_idx],
                &flux_z_d[gpu][stream_idx], &wavefront_length_d[gpu][corner],
                &wavefront_offset_d[gpu][corner], &wavefront_x_d[gpu][corner],
                &wavefront_y_d[gpu][corner], &wavefront_z_d[gpu][corner],
                &mutex_in_d[gpu][stream_idx], &mutex_out_d[gpu][stream_idx] };
      switch (angles_per_thread)
      {
        case 1:
          {
            if (grid.x == 0)
            {
              grid.x = compute_blocks_per_sweep(x_range, y_range, z_range,            
                  (const void*)gpu_time_independent_sweep_without_fixup<1>, 
                  threads_per_block, runtime, ctx, gpu);
              blocks_per_sweep[gpu] = grid.x;
            }
            cudaLaunchCooperativeKernel((const void*)gpu_time_independent_sweep_without_fixup<1>,
                grid, block, (void**)args, 0, stream);
            break;
          }
        case 2:
          {
            if (grid.x == 0)
            {
              grid.x = compute_blocks_per_sweep(x_range, y_range, z_range,            
                  (const void*)gpu_time_independent_sweep_without_fixup<1>, 
                  threads_per_block, runtime, ctx, gpu);
              blocks_per_sweep[gpu] = grid.x;
            }
            cudaLaunchCooperativeKernel((const void*)gpu_time_independent_sweep_without_fixup<2>,
                grid, block, (void**)args, 0, stream);
            break;
          }
        default:
          printf("ON SNAP! That is a lot of angles! Add more cases!\n");
          exit(1);
      }
    }
  }
  cudaStreamDestroy(stream);
}

