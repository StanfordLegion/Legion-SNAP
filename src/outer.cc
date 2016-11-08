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
#include "outer.h"

using namespace LegionRuntime::Accessor;

//------------------------------------------------------------------------------
CalcOuterSource::CalcOuterSource(const Snap &snap, const Predicate &pred,
                                 const SnapArray &qi, const SnapArray &slgg,
                                 const SnapArray &mat, const SnapArray &q2rgp0, 
                                 const SnapArray &q2grpm)
  : SnapTask<CalcOuterSource>(snap, snap.get_launch_bounds(), pred)
//------------------------------------------------------------------------------
{
  qi.add_projection_requirement(READ_ONLY, *this); // qi0
  slgg.add_region_requirement(READ_ONLY, *this); // sxs_g
  mat.add_projection_requirement(READ_ONLY, *this); // map
  q2rgp0.add_projection_requirement(WRITE_DISCARD, *this); // qo0
  // Only have to initialize this if there are multiple moments
  if (Snap::num_moments > 1)
    q2grpm.add_projection_requirement(WRITE_DISCARD, *this); // qom
  else
    q2grpm.initialize();
}

//------------------------------------------------------------------------------
/*static*/ void CalcOuterSource::preregister_cpu_variants(void)
//------------------------------------------------------------------------------
{
  register_cpu_variant<cpu_implementation>(true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ void CalcOuterSource::preregister_gpu_variants(void)
//------------------------------------------------------------------------------
{
  register_gpu_variant<gpu_implementation>(true/*leaf*/);
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
  // TODO: Implement GPU kernels
  assert(false);
}

//------------------------------------------------------------------------------
TestOuterConvergence::TestOuterConvergence(const Snap &snap, 
                                           const Predicate &pred,
                                           const SnapArray &flux0,
                                           const SnapArray &flux0po,
                                           const Future &inner_converged,
                                           const Future &true_future)
  : SnapTask<TestOuterConvergence>(snap, snap.get_launch_bounds(), pred)
//------------------------------------------------------------------------------
{
  flux0.add_projection_requirement(READ_ONLY, *this);
  flux0po.add_projection_requirement(READ_ONLY, *this);
  add_future(inner_converged);
  predicate_false_future = true_future;
}

//------------------------------------------------------------------------------
/*static*/ void TestOuterConvergence::preregister_cpu_variants(void)
//------------------------------------------------------------------------------
{
  register_cpu_variant<bool, cpu_implementation>(true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ void TestOuterConvergence::preregister_gpu_variants(void)
//------------------------------------------------------------------------------
{
  register_gpu_variant<bool, gpu_implementation>(true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ bool TestOuterConvergence::cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
  // If the inner loop didn't converge, then we can't either
  assert(!task->futures.empty());
  bool inner_converged = task->futures[0].get_result<bool>();
  if (!inner_converged)
    return false;
  // Get the index space domain for iteration
  assert(task->regions[0].region.get_index_space() == 
         task->regions[1].region.get_index_space());
  Domain dom = runtime->get_index_space_domain(ctx, 
          task->regions[0].region.get_index_space());
  Rect<3> subgrid_bounds = dom.get_rect<3>();
  const double tolr = 1.0e-12;
  const double epsi = 100.0 * Snap::convergence_eps;
  // Iterate over all the energy groups
  assert(task->regions[0].privilege_fields.size() == 
         task->regions[1].privilege_fields.size());
  for (std::set<FieldID>::const_iterator it = 
        task->regions[0].privilege_fields.begin(); it !=
        task->regions[0].privilege_fields.end(); it++)
  {
    RegionAccessor<AccessorType::Generic,double> fa_flux0 = 
      regions[0].get_field_accessor(*it).typeify<double>();
    RegionAccessor<AccessorType::Generic,double> fa_flux0po = 
      regions[1].get_field_accessor(*it).typeify<double>();
    for (GenericPointInRectIterator<3> itr(subgrid_bounds); itr; itr++)
    {
      DomainPoint dp = DomainPoint::from_point<3>(itr.p);
      double flux0po = fa_flux0po.read(dp);
      double df = 1.0;
      if (fabs(flux0po) < tolr) {
        flux0po = 1.0;
        df = 0.0;
      }
      double flux0 = fa_flux0.read(dp);
      df = fabs( (flux0 / flux0po) - df );
      // Skip anything less than -INF
      if (df < -INFINITY)
        continue;
      if (df > epsi)
        return false;
    }
  }
  return true;
}

//------------------------------------------------------------------------------
/*static*/ bool TestOuterConvergence::gpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
  // TODO: Implement GPU kernels
  assert(false);
  return false;
}

