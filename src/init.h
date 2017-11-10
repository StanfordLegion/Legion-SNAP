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

#ifndef __INIT_H__
#define __INIT_H__

#include "snap.h"
#include "legion.h"

class InitMaterial : public SnapTask<InitMaterial,
                                     Snap::INIT_MATERIAL_TASK_ID> {
public:
  InitMaterial(const Snap &snap, const SnapArray<3> &mat);
public:
  static void preregister_cpu_variants(void);
public:
  static void cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
};

class InitSource : public SnapTask<InitSource, Snap::INIT_SOURCE_TASK_ID> {
public:
  InitSource(const Snap &snap, const SnapArray<3> &qi);
public:
  static void preregister_cpu_variants(void);
public:
  static void cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
};

class InitScattering : public TaskLauncher {
public:
  static const Snap::SnapTaskID TASK_ID = Snap::INIT_SCATTERING_TASK_ID;
public:
  InitScattering(const SnapArray<1> &sigt,
                 const SnapArray<1> &siga,
                 const SnapArray<1> &sigs,
                 const SnapArray<2> &slgg);
public:
  void dispatch(Context ctx, Runtime *runtime);
public:
  static void preregister_cpu_variants(void);
public:
  static void cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
};

class InitVelocity : public TaskLauncher {
public:
  static const Snap::SnapTaskID TASK_ID = Snap::INIT_VELOCITY_TASK_ID;
public:
  InitVelocity(const SnapArray<1> &vel,
               const SnapArray<1> &vdelt);
public:
  void dispatch(Context ctx, Runtime *runtime);
public:
  static void preregister_cpu_variants(void);
public:
  static void cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
};

class InitGPUSweep : public SnapTask<InitGPUSweep, Snap::INIT_GPU_SWEEP_TASK_ID> {
public:
  InitGPUSweep(const Snap &snap, const Rect<3> &launch_bounds); 
public:
  static void preregister_gpu_variants(void);
public:
  static void gpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
};

#endif // __INIT_H__

