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
#include "inner.h"

//------------------------------------------------------------------------------
CalcInnerSource::CalcInnerSource(const Snap &snap, const Predicate &pred,
                               const SnapArray &s_xs, const SnapArray &flux0,
                               const SnapArray &fluxm, const SnapArray &q2grp0,
                               const SnapArray &q2grpm, const SnapArray &qtot)
  : SnapTask<CalcInnerSource>(snap, snap.get_launch_bounds(), pred)
//------------------------------------------------------------------------------
{
  s_xs.add_projection_requirement(READ_ONLY, *this);
  flux0.add_projection_requirement(READ_ONLY, *this);
  q2grp0.add_projection_requirement(READ_ONLY, *this);
  q2grpm.add_projection_requirement(READ_ONLY, *this);
  qtot.add_projection_requirement(WRITE_DISCARD, *this);
  // only include this requirement if we have more than one moment
  if (Snap::num_moments > 1)
    fluxm.add_projection_requirement(READ_ONLY, *this);
}

//------------------------------------------------------------------------------
/*static*/ void CalcInnerSource::preregister_cpu_variants(void)
//------------------------------------------------------------------------------
{
  register_cpu_variant<cpu_implementation>(true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ void CalcInnerSource::preregister_gpu_variants(void)
//------------------------------------------------------------------------------
{
  register_gpu_variant<gpu_implementation>(true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ void CalcInnerSource::cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{

}

//------------------------------------------------------------------------------
/*static*/ void CalcInnerSource::gpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{

}

//------------------------------------------------------------------------------
TestInnerConvergence::TestInnerConvergence(const Snap &snap, 
                                           const Predicate &pred,
                                           const SnapArray &flux0,
                                           const SnapArray &flux0pi)
  : SnapTask<TestInnerConvergence>(snap, snap.get_launch_bounds(), pred)
//------------------------------------------------------------------------------
{
  flux0.add_projection_requirement(READ_ONLY, *this);
  flux0pi.add_projection_requirement(READ_ONLY, *this);
}

//------------------------------------------------------------------------------
/*static*/ void TestInnerConvergence::preregister_cpu_variants(void)
//------------------------------------------------------------------------------
{
  register_cpu_variant<bool, cpu_implementation>(true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ void TestInnerConvergence::preregister_gpu_variants(void)
//------------------------------------------------------------------------------
{
  register_gpu_variant<bool, gpu_implementation>(true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ bool TestInnerConvergence::cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
  return false;
}

//------------------------------------------------------------------------------
/*static*/ bool TestInnerConvergence::gpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
  return false;
}

