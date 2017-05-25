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

#ifndef __SWEEP_H__
#define __SWEEP_H__

#include "snap.h"
#include "legion.h"

class MiniKBATask : public SnapTask<MiniKBATask, Snap::MINI_KBA_TASK_ID, 2> {
public:
  static const int NON_GHOST_REQUIREMENTS = 3;
public:
  struct MiniKBAArgs {
  public:
    MiniKBAArgs(int c, int start, int stop)
      : wavefront(0), corner(c), group_start(start), group_stop(stop) { }
  public:
    int wavefront;
    int corner;
    int group_start;
    int group_stop; // inclusive
  };
public:
  MiniKBATask(const Snap &snap, const Predicate &pred, 
              const SnapArray<3> &flux, const SnapArray<3> &fluxm,
              const SnapArray<3> &qtot, const SnapArray<1> &vdelt, 
              const SnapArray<3> &dinv, const SnapArray<3> &t_xs, 
              const SnapArray<3> &time_flux_in, 
              const SnapArray<3> &time_flux_out,
              const SnapArray<3> &qim, const SnapArray<2> &flux_xy,
              const SnapArray<2> &flux_yz, const SnapArray<2> &flux_xz,
              int group_start, int group_stop, int corner, 
              const int ghost_offsets[3]);
public:
  void dispatch_wavefront(int wavefront, IndexSpace<2> launch_sp, 
                          Context cxt, Runtime *runtime);
public:
  MiniKBAArgs mini_kba_args;
public:
  static void preregister_cpu_variants(void);
  static void preregister_gpu_variants(void);
public:
  static void cpu_implementation(const Task *task,
     const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
  static void sse_implementation(const Task *task, Context ctx, Runtime *runtime,
     const std::vector<MomentQuad*> &qtot_ptrs, const ByteOffset qtot_offsets[3],
     const std::vector<double*> &flux_ptrs, const ByteOffset flux_offsets[3],
     const std::vector<double*> &qim_ptrs, const ByteOffset qim_offsets[3],
     const std::vector<MomentTriple*> &fluxm_ptrs, const ByteOffset fluxm_offsets[3],
     const std::vector<double*> &dinv_ptrs, const ByteOffset dinv_offsets[3],
     const std::vector<double*> &time_flux_in_ptrs, const ByteOffset time_flux_in_offsets[3],
     const std::vector<double*> &time_flux_out_ptrs, const ByteOffset time_flux_out_offsets[3],
     const std::vector<double*> &t_xs_ptrs, const ByteOffset t_xs_offsets[3],
     const std::vector<double*> &ghost_x_ptrs, const ByteOffset ghostx_offsets[2],
     const std::vector<double*> &ghost_y_ptrs, const ByteOffset ghosty_offsets[2],
     const std::vector<double*> &ghost_z_ptrs, const ByteOffset ghostz_offsets[2],
     const std::vector<double*> &vdelt_ptrs, const ByteOffset vdelt_offsets[1]);
  static void avx_implementation(const Task *task, Context ctx, Runtime *runtime,
     const std::vector<MomentQuad*> &qtot_ptrs, const ByteOffset qtot_offsets[3],
     const std::vector<double*> &flux_ptrs, const ByteOffset flux_offsets[3],
     const std::vector<double*> &qim_ptrs, const ByteOffset qim_offsets[3],
     const std::vector<MomentTriple*> &fluxm_ptrs, const ByteOffset fluxm_offsets[3],
     const std::vector<double*> &dinv_ptrs, const ByteOffset dinv_offsets[3],
     const std::vector<double*> &time_flux_in_ptrs, const ByteOffset time_flux_in_offsets[3],
     const std::vector<double*> &time_flux_out_ptrs, const ByteOffset time_flux_out_offsets[3],
     const std::vector<double*> &t_xs_ptrs, const ByteOffset t_xs_offsets[3],
     const std::vector<double*> &ghost_x_ptrs, const ByteOffset ghostx_offsets[2],
     const std::vector<double*> &ghost_y_ptrs, const ByteOffset ghosty_offsets[2],
     const std::vector<double*> &ghost_z_ptrs, const ByteOffset ghostz_offsets[2],
     const std::vector<double*> &vdelt_ptrs, const ByteOffset vdelt_offsets[1]);
  static void gpu_implementation(const Task *task, Context ctx, Runtime *runtime,
     const std::vector<MomentQuad*> &qtot_ptrs, const ByteOffset qtot_offsets[3],
     const std::vector<double*> &flux_ptrs, const ByteOffset flux_offsets[3],
     const std::vector<double*> &qim_ptrs, const ByteOffset qim_offsets[3],
     const std::vector<MomentTriple*> &fluxm_ptrs, const ByteOffset fluxm_offsets[3],
     const std::vector<double*> &dinv_ptrs, const ByteOffset dinv_offsets[3],
     const std::vector<double*> &time_flux_in_ptrs, const ByteOffset time_flux_in_offsets[3],
     const std::vector<double*> &time_flux_out_ptrs, const ByteOffset time_flux_out_offsets[3],
     const std::vector<double*> &t_xs_ptrs, const ByteOffset t_xs_offsets[3],
     const std::vector<double*> &ghost_x_ptrs, const ByteOffset ghostx_offsets[2],
     const std::vector<double*> &ghost_y_ptrs, const ByteOffset ghosty_offsets[2],
     const std::vector<double*> &ghost_z_ptrs, const ByteOffset ghostz_offsets[2],
     const std::vector<double*> &vdelt_ptrs, const ByteOffset vdelt_offsets[1]);
};

#endif // __SWEEP_H__

