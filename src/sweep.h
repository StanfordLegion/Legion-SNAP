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

class MiniKBATask : public SnapTask<MiniKBATask> {
public:
  static const int NON_GHOST_REQUIREMENTS = 3;
public:
  struct MiniKBAArgs {
  public:
    MiniKBAArgs(int c, int g)
      : wavefront(0), corner(c), group(g) { }
  public:
    int wavefront;
    int corner;
    int group;
  };
public:
  static const Snap::SnapTaskID TASK_ID = Snap::MINI_KBA_TASK_ID;
  static const Snap::SnapReductionID REDOP = Snap::NO_REDUCTION_ID;
public:
  MiniKBATask(const Snap &snap, const Predicate &pred, bool even, 
              const SnapArray &flux, const SnapArray &fluxm,
              const SnapArray &qtot, const SnapArray &vdelt, 
              const SnapArray &dinv, const SnapArray &t_xs, 
              const SnapArray &time_flux_in, 
              const SnapArray &time_flux_out,
              int group, int corner, const int ghost_offsets[3]);
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
  static void gpu_implementation(const Task *task,
     const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
};

#endif // __SWEEP_H__

