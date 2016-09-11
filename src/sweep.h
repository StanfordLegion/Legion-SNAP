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
  static const Snap::SnapTaskID TASK_ID = Snap::MINI_KBA_TASK_ID;
  static const Snap::SnapReductionID REDOP = Snap::NO_REDUCTION_ID;
public:
  MiniKBATask(const Snap &snap, const Predicate &pred, 
              const SnapArray &flux, const SnapArray &qtot,
              int group, int corner, bool even);
public:
  void dispatch_wavefront(const Domain &launch_domain, 
                          Context cxt, Runtime *runtime);
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

