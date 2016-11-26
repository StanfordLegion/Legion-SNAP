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

#include "snap.h"
#include "mms.h"

using namespace LegionRuntime::Accessor;

//------------------------------------------------------------------------------
MMSInit::MMSInit(const Snap &snap, const SnapArray &qim, int c)
  : SnapTask<MMSInit>(snap, snap.get_launch_bounds(), Predicate::TRUE_PRED),
    corner(c)
//------------------------------------------------------------------------------
{
  global_arg = TaskArgument(&corner, sizeof(corner));
  qim.add_projection_requirement(READ_WRITE, *this);
}

//------------------------------------------------------------------------------
/*static*/ void MMSInit::preregister_cpu_variants(void)
//------------------------------------------------------------------------------
{
  register_cpu_variant<cpu_implementation>(true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ void MMSInit::cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE

#endif
}

//------------------------------------------------------------------------------
MMSScale::MMSScale(const Snap &snap, const SnapArray &qim, double f)
  : SnapTask<MMSScale>(snap, snap.get_launch_bounds(), Predicate::TRUE_PRED),
    scale_factor(f)
//------------------------------------------------------------------------------
{
  global_arg = TaskArgument(&scale_factor, sizeof(scale_factor));
  qim.add_projection_requirement(READ_WRITE, *this);
}

//------------------------------------------------------------------------------
/*static*/ void MMSScale::preregister_cpu_variants(void)
//------------------------------------------------------------------------------
{
  register_cpu_variant<cpu_implementation>(true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ void MMSScale::cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE

#endif
}

//------------------------------------------------------------------------------
MMSVerify::MMSVerify(const Snap &snap, const SnapArray &flux)
  : SnapTask<MMSVerify>(snap, snap.get_launch_bounds(), Predicate::TRUE_PRED)
//------------------------------------------------------------------------------
{
  flux.add_projection_requirement(READ_ONLY, *this);
}

//------------------------------------------------------------------------------
/*static*/ void MMSVerify::preregister_cpu_variants(void)
//------------------------------------------------------------------------------
{
  register_cpu_variant<cpu_implementation>(true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ void MMSVerify::cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE

#endif
}

