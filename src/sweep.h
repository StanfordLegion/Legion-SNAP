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

#ifndef __SWEEP_H__
#define __SWEEP_H__

#include "snap.h"
#include "legion.h"

using namespace Legion;

class MiniKBATask : public SnapTask<MiniKBATask, Snap::MINI_KBA_TASK_ID> {
public:
  static const int NON_GHOST_REQUIREMENTS = 3;
public:
  struct MiniKBAArgs {
  public:
    MiniKBAArgs(int c, int start, int stop, bool e)
      : wavefront(0), corner(c), group_start(start), 
        group_stop(stop), even(e) { }
  public:
    int wavefront;
    int corner;
    int group_start;
    int group_stop; // inclusive
    bool even;
  };
public:
  MiniKBATask(const Snap &snap, const Predicate &pred, bool even, 
              const SnapArray &flux, const SnapArray &fluxm,
              const SnapArray &qtot, const SnapArray &vdelt, 
              const SnapArray &dinv, const SnapArray &t_xs, 
              const SnapArray &time_flux_in, 
              const SnapArray &time_flux_out,
              const SnapArray &qim, int group_start,
              int group_stop, int corner, 
              const int ghost_offsets[3]);
public:
  void dispatch_wavefront(int wavefront, const Domain &launch_domain, 
                          Context cxt, Runtime *runtime);
public:
  MiniKBAArgs mini_kba_args;
public:
  static void preregister_cpu_variants(void);
  static void preregister_gpu_variants(void);
public:
  static void cpu_implementation(const Task *task,
     const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
  static void sse_implementation(const Task *task,
     const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
  static void avx_implementation(const Task *task,
     const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
  static void gpu_implementation(const Task *task,
     const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
};

#endif // __SWEEP_H__

