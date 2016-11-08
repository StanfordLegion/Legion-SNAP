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

#include <cmath>

#include "snap.h"
#include "inner.h"

using namespace LegionRuntime::Accessor;

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
                                           const SnapArray &flux0pi,
                                           const Future &true_future)
  : SnapTask<TestInnerConvergence>(snap, snap.get_launch_bounds(), pred)
//------------------------------------------------------------------------------
{
  flux0.add_projection_requirement(READ_ONLY, *this);
  flux0pi.add_projection_requirement(READ_ONLY, *this);
  predicate_false_future = true_future;
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
  // Get the index space domain for iteration
  assert(task->regions[0].region.get_index_space() == 
         task->regions[1].region.get_index_space());
  Domain dom = runtime->get_index_space_domain(ctx, 
          task->regions[0].region.get_index_space());
  Rect<3> subgrid_bounds = dom.get_rect<3>();
  const double tolr = 1.0e-12;
  const double epsi = Snap::convergence_eps;
  // Iterate over all the energy groups
  assert(task->regions[0].privilege_fields.size() == 
         task->regions[1].privilege_fields.size());
  for (std::set<FieldID>::const_iterator it = 
        task->regions[0].privilege_fields.begin(); it !=
        task->regions[0].privilege_fields.end(); it++)
  {
    RegionAccessor<AccessorType::Generic,double> fa_flux0 = 
      regions[0].get_field_accessor(*it).typeify<double>();
    RegionAccessor<AccessorType::Generic,double> fa_flux0pi = 
      regions[1].get_field_accessor(*it).typeify<double>();
    for (GenericPointInRectIterator<3> itr(subgrid_bounds); itr; itr++)
    {
      DomainPoint dp = DomainPoint::from_point<3>(itr.p);
      double flux0pi = fa_flux0pi.read(dp);
      double df = 1.0;
      if (fabs(flux0pi) < tolr) {
        flux0pi = 1.0;
        df = 0.0;
      }
      double flux0 = fa_flux0.read(dp);
      df = fabs( (flux0 / flux0pi) - df );
      if (df > epsi)
        return false;
    }
  }
  return true;
}

//------------------------------------------------------------------------------
/*static*/ bool TestInnerConvergence::gpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
  assert(false);
  return false;
}

