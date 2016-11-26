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

#ifndef __MMS_H__
#define __MMS_H__

#include "snap.h"
#include "legion.h"

class MMSInit : public SnapTask<MMSInit, Snap::MMS_INIT_TASK_ID> {
public:
  MMSInit(const Snap &snap, const SnapArray &qim, int corner);
public:
  const int corner;
public:
  static void preregister_cpu_variants(void);
public:
  static void cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
};

class MMSScale : public SnapTask<MMSScale, Snap::MMS_SCALE_TASK_ID> {
public:
  MMSScale(const Snap &snap, const SnapArray &qim, double factor);
public:
  const double scale_factor;
public:
  static void preregister_cpu_variants(void);
public:
  static void cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
};

class MMSVerify : public SnapTask<MMSVerify, Snap::MMS_VERIFY_TASK_ID> {
public:
  MMSVerify(const Snap &snap, const SnapArray &flux);
public:
  static void preregister_cpu_variants(void);
public:
  static void cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
};

#endif // __MMS_H__

