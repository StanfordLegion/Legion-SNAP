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
#include "outer.h"

//------------------------------------------------------------------------------
CalcOuterSource::CalcOuterSource(const Snap &snap, const Predicate &pred,
                                 const SnapArray &qi, const SnapArray &slgg,
                                 const SnapArray &mat, const SnapArray &q2rgp0, 
                                 const SnapArray &q2grpm)
  : SnapTask<CalcOuterSource>(snap, snap.get_launch_bounds(), pred)
//------------------------------------------------------------------------------
{
  qi.add_projection_requirement(READ_ONLY, *this);
  slgg.add_region_requirement(READ_ONLY, *this);
  mat.add_projection_requirement(READ_ONLY, *this);
  q2rgp0.add_projection_requirement(WRITE_DISCARD, *this);
  q2grpm.add_projection_requirement(WRITE_DISCARD, *this);
}

//------------------------------------------------------------------------------
/*static*/ void CalcOuterSource::preregister_cpu_variants(void)
//------------------------------------------------------------------------------
{
  register_cpu_variant<cpu_implementation>();
}

//------------------------------------------------------------------------------
/*static*/ void CalcOuterSource::preregister_gpu_variants(void)
//------------------------------------------------------------------------------
{
  register_gpu_variant<gpu_implementation>();
}

//------------------------------------------------------------------------------
/*static*/ void CalcOuterSource::cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{

}

//------------------------------------------------------------------------------
/*static*/ void CalcOuterSource::gpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{

}

//------------------------------------------------------------------------------
TestOuterConvergence::TestOuterConvergence(const Snap &snap, 
                                           const Predicate &pred,
                                           const SnapArray &flux0,
                                           const SnapArray &flux0po)
  : SnapTask<TestOuterConvergence>(snap, snap.get_launch_bounds(), pred)
//------------------------------------------------------------------------------
{
  flux0.add_projection_requirement(READ_ONLY, *this);
  flux0po.add_projection_requirement(READ_ONLY, *this);
}

//------------------------------------------------------------------------------
/*static*/ void TestOuterConvergence::preregister_cpu_variants(void)
//------------------------------------------------------------------------------
{
  register_cpu_variant<bool, cpu_implementation>();
}

//------------------------------------------------------------------------------
/*static*/ void TestOuterConvergence::preregister_gpu_variants(void)
//------------------------------------------------------------------------------
{
  register_gpu_variant<bool, gpu_implementation>();
}

//------------------------------------------------------------------------------
/*static*/ bool TestOuterConvergence::cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
  return false;
}

//------------------------------------------------------------------------------
/*static*/ bool TestOuterConvergence::gpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
  return false;
}

