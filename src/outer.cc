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
                                 const SnapArray &q2grpm, 
                                 const SnapArray &flux0, const SnapArray &fluxm)
  : SnapTask<CalcOuterSource, Snap::CALC_OUTER_SOURCE_TASK_ID>(
      snap, snap.get_launch_bounds(), pred)
//------------------------------------------------------------------------------
{
  qi.add_projection_requirement(READ_ONLY, *this); // qi0
  flux0.add_projection_requirement(READ_ONLY, *this); // flux0
  slgg.add_region_requirement(READ_ONLY, *this); // sxs_g
  mat.add_projection_requirement(READ_ONLY, *this); // map
  q2rgp0.add_projection_requirement(WRITE_DISCARD, *this); // qo0
  // Only have to initialize this if there are multiple moments
  if (Snap::num_moments > 1) {
    fluxm.add_projection_requirement(READ_ONLY, *this); // fluxm 
    q2grpm.add_projection_requirement(WRITE_DISCARD, *this); // qom
  }
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
#ifndef NO_COMPUTE
  Domain dom = runtime->get_index_space_domain(ctx, 
          task->regions[0].region.get_index_space());
  Rect<3> subgrid_bounds = dom.get_rect<3>();
  const bool multi_moment = (Snap::num_moments > 1);
  const unsigned num_groups = task->regions[0].privilege_fields.size();
  assert(num_groups == task->regions[1].privilege_fields.size());
  assert(num_groups == task->regions[4].privilege_fields.size());
  // Make the accessors for all the groups up front
  std::vector<RegionAccessor<AccessorType::Generic,double> > fa_qi0(num_groups);
  std::vector<RegionAccessor<AccessorType::Generic,double> > fa_flux0(num_groups);
  std::vector<RegionAccessor<AccessorType::Generic,double> > fa_qo0(num_groups);
  std::vector<RegionAccessor<AccessorType::Generic,MomentTriple> > 
    fa_fluxm(multi_moment ? num_groups : 0);
  // Field spaces are all the same so this is safe
  unsigned g = 0;
  for (std::set<FieldID>::const_iterator it = 
        task->regions[0].privilege_fields.begin(); it !=
        task->regions[0].privilege_fields.end(); it++, g++)
  {
    fa_qi0[g] = regions[0].get_field_accessor(*it).typeify<double>();
    fa_flux0[g] = regions[1].get_field_accessor(*it).typeify<double>();
    fa_qo0[g] = regions[4].get_field_accessor(*it).typeify<double>();
    if (multi_moment)
      fa_fluxm[g] = regions[5].get_field_accessor(*it).typeify<MomentTriple>();
  }
  RegionAccessor<AccessorType::Generic,int> fa_mat = 
    regions[3].get_field_accessor(Snap::FID_SINGLE).typeify<int>();
  // Do pairwise group intersections
  g = 0;
  for (std::set<FieldID>::const_iterator it = 
        task->regions[0].privilege_fields.begin(); it !=
        task->regions[0].privilege_fields.end(); it++, g++)
  {
    RegionAccessor<AccessorType::Generic,MomentQuad> fa_slgg = 
      regions[2].get_field_accessor(*it).typeify<MomentQuad>();
    for (GenericPointInRectIterator<3> itr(subgrid_bounds); itr; itr++)
    {
      DomainPoint dp = DomainPoint::from_point<3>(itr.p);
      double qo0 = fa_qi0[g].read(dp);
      for (unsigned g2 = 0; g2 < num_groups; g2++)
      {
        if (g == g2)
          continue;
        const int mat = fa_mat.read(dp);
        const int pvals[2] = { mat, g2 }; 
        DomainPoint xsp = DomainPoint::from_point<2>(Point<2>(pvals));
        const MomentQuad cs = fa_slgg.read(xsp);
        qo0 += cs[0] * fa_flux0[g2].read(dp);
      }
      fa_qo0[g].write(dp, qo0);
    }
    if (multi_moment)
    {
      RegionAccessor<AccessorType::Generic,MomentTriple> fa_qom = 
        regions[6].get_field_accessor(*it).typeify<MomentTriple>();
      for (GenericPointInRectIterator<3> itr(subgrid_bounds); itr; itr++)
      {
        DomainPoint dp = DomainPoint::from_point<3>(itr.p);
        MomentTriple qom;
        for (unsigned g2 = 0; g2 < num_groups; g2++)
        {
          if (g == g2)
            continue;
          const int mat = fa_mat.read(dp);
          int moment = 0;
          MomentTriple csm;
          for (int l = 1; l < Snap::num_moments; l++)
          {
            const int pvals[2] = { mat, g2 };
            DomainPoint xsp = DomainPoint::from_point<2>(Point<2>(pvals));
            const MomentQuad scat = fa_slgg.read(xsp);
            for (int i = 0; i < Snap::lma[l]; i++)
              csm[moment+i] = scat[l];
            moment += Snap::lma[l];
          }
          MomentTriple fluxm = fa_fluxm[g2].read(dp); 
          for (int i = 0; i < (Snap::num_moments-1); i++)
            qom[i] += csm[i] * fluxm[i];
        }
        fa_qom.write(dp, qom);  
      }
    }
  }
#endif
}

//------------------------------------------------------------------------------
/*static*/ void CalcOuterSource::gpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  // TODO: Implement GPU kernels
  assert(false);
#endif
}

//------------------------------------------------------------------------------
TestOuterConvergence::TestOuterConvergence(const Snap &snap, 
                                           const Predicate &pred,
                                           const SnapArray &flux0,
                                           const SnapArray &flux0po,
                                           const Future &inner_converged,
                                           const Future &true_future)
  : SnapTask<TestOuterConvergence, Snap::TEST_OUTER_CONVERGENCE_TASK_ID>(
      snap, snap.get_launch_bounds(), pred)
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
#ifndef NO_COMPUTE
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
#else
  return false;
#endif
}

//------------------------------------------------------------------------------
/*static*/ bool TestOuterConvergence::gpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  // TODO: Implement GPU kernels
  assert(false);
#endif
  return false;
}

