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
CalcOuterSource::CalcOuterSource(const Snap &snap, const Predicate &pred)
  : SnapTask<CalcOuterSource>(snap, snap.get_launch_bounds(), pred)
//------------------------------------------------------------------------------
{
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
                                           const Predicate &pred)
  : SnapTask<TestOuterConvergence>(snap, snap.get_launch_bounds(), pred)
//------------------------------------------------------------------------------
{
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

