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

#ifndef __INNER_H__
#define __INNER_H__

#include "snap.h"
#include "legion.h"

using namespace Legion;

class CalcInnerSource : public SnapTask<CalcInnerSource, 
                                        Snap::CALC_INNER_SOURCE_TASK_ID> {
public:
  CalcInnerSource(const Snap &snap, const Predicate &pred,
                  const SnapArray &s_xs, const SnapArray &flux0,
                  const SnapArray &fluxm, const SnapArray &q2grp0,
                  const SnapArray &q2grpm, const SnapArray &qtot);
public:
  static void preregister_cpu_variants(void);
  static void preregister_gpu_variants(void);
public:
  static void cpu_implementation(const Task *task,
     const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
  static void fast_implementation(const Task *task,
     const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
  static void gpu_implementation(const Task *task,
     const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
};

class TestInnerConvergence : public SnapTask<TestInnerConvergence,
                                             Snap::TEST_INNER_CONVERGENCE_TASK_ID> {
public:
  TestInnerConvergence(const Snap &snap, const Predicate &pred,
                       const SnapArray &flux0, const SnapArray &flux0pi,
                       const Future &true_future, int group_start, int group_stop);
public:
  static void preregister_cpu_variants(void);
  static void preregister_gpu_variants(void);
public:
  static bool cpu_implementation(const Task *task,
     const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
  static bool fast_implementation(const Task *task,
     const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
  static bool gpu_implementation(const Task *task,
     const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
};

#endif // __INNER_H__
