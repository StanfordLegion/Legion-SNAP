/* Copyright 2017 NVIDIA Corporation
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

extern LegionRuntime::Logger::Category log_snap;

using namespace LegionRuntime::Accessor;

//------------------------------------------------------------------------------
CalcInnerSource::CalcInnerSource(const Snap &snap, const Predicate &pred,
                               const SnapArray &s_xs, const SnapArray &flux0,
                               const SnapArray &fluxm, const SnapArray &q2grp0,
                               const SnapArray &q2grpm, const SnapArray &qtot)
  : SnapTask<CalcInnerSource, Snap::CALC_INNER_SOURCE_TASK_ID>(
      snap, snap.get_launch_bounds(), pred)
//------------------------------------------------------------------------------
{
  s_xs.add_projection_requirement(READ_ONLY, *this);
  flux0.add_projection_requirement(READ_ONLY, *this);
  q2grp0.add_projection_requirement(READ_ONLY, *this);
  qtot.add_projection_requirement(WRITE_DISCARD, *this);
  // only include this requirement if we have more than one moment
  if (Snap::num_moments > 1) {
    fluxm.add_projection_requirement(READ_ONLY, *this);
    q2grpm.add_projection_requirement(READ_ONLY, *this);
  }
}

//------------------------------------------------------------------------------
/*static*/ void CalcInnerSource::preregister_cpu_variants(void)
//------------------------------------------------------------------------------
{
  register_cpu_variant<fast_implementation>(true/*leaf*/);
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
#ifndef NO_COMPUTE
  log_snap.info("Running Calc Inner Source");

  Domain dom = runtime->get_index_space_domain(ctx, 
          task->regions[0].region.get_index_space());
  Rect<3> subgrid_bounds = dom.get_rect<3>();
  const bool multi_moment = (Snap::num_moments > 1);
  const unsigned num_groups = task->regions[0].privilege_fields.size();
  assert(num_groups == task->regions[1].privilege_fields.size());
  assert(num_groups == task->regions[2].privilege_fields.size());
  for (std::set<FieldID>::const_iterator it = 
        task->regions[0].privilege_fields.begin(); it !=
        task->regions[0].privilege_fields.end(); it++)
  {
    RegionAccessor<AccessorType::Generic,MomentQuad> fa_sxs = 
      regions[0].get_field_accessor(*it).typeify<MomentQuad>();
    RegionAccessor<AccessorType::Generic,double> fa_flux0 = 
      regions[1].get_field_accessor(*it).typeify<double>();
    RegionAccessor<AccessorType::Generic,double> fa_q2grp0 = 
      regions[2].get_field_accessor(*it).typeify<double>();
    RegionAccessor<AccessorType::Generic,MomentQuad> fa_qtot = 
      regions[3].get_field_accessor(*it).typeify<MomentQuad>();
    if (multi_moment) {
      RegionAccessor<AccessorType::Generic,MomentTriple> fa_fluxm = 
        regions[4].get_field_accessor(*it).typeify<MomentTriple>();
      RegionAccessor<AccessorType::Generic,MomentTriple> fa_q2grpm = 
        regions[5].get_field_accessor(*it).typeify<MomentTriple>();
      for (GenericPointInRectIterator<3> itr(subgrid_bounds); itr; itr++)
      {
        DomainPoint dp = DomainPoint::from_point<3>(itr.p);
        MomentQuad sxs_quad = fa_sxs.read(dp);
        const double q0 = fa_q2grp0.read(dp);
        const double flux0 = fa_flux0.read(dp);
        MomentQuad quad;
        quad[0] = q0 + flux0 * sxs_quad[0];
        MomentTriple qom = fa_q2grpm.read(dp);
        MomentTriple fm = fa_fluxm.read(dp);
        int moment = 0;
        for (int l = 1; l < Snap::num_moments; l++) {
          for (int i = 0; i < Snap::lma[l]; i++)
            quad[moment+i+1] = qom[moment+i] + fm[moment+i] * sxs_quad[l];
          moment += Snap::lma[l];
        }
        fa_qtot.write(dp, quad);
      }
    } else {
      for (GenericPointInRectIterator<3> itr(subgrid_bounds); itr; itr++)
      {
        DomainPoint dp = DomainPoint::from_point<3>(itr.p);
        MomentQuad sxs_quad = fa_sxs.read(dp);
        const double q0 = fa_q2grp0.read(dp);
        const double flux0 = fa_flux0.read(dp);
        MomentQuad quad;
        quad[0] = q0 + flux0 * sxs_quad[0];
        fa_qtot.write(dp, quad);
      }
    }
  }
#endif
}

//------------------------------------------------------------------------------
/*static*/ void CalcInnerSource::fast_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  log_snap.info("Running Fast Calc Inner Source");

  Domain dom = runtime->get_index_space_domain(ctx, 
          task->regions[0].region.get_index_space());
  Rect<3> subgrid_bounds = dom.get_rect<3>();
  const bool multi_moment = (Snap::num_moments > 1);
  const unsigned num_groups = task->regions[0].privilege_fields.size();
  assert(num_groups == task->regions[1].privilege_fields.size());
  assert(num_groups == task->regions[2].privilege_fields.size());
  for (std::set<FieldID>::const_iterator it = 
        task->regions[0].privilege_fields.begin(); it !=
        task->regions[0].privilege_fields.end(); it++)
  {
    RegionAccessor<AccessorType::Generic,MomentQuad> fa_sxs = 
      regions[0].get_field_accessor(*it).typeify<MomentQuad>();
    RegionAccessor<AccessorType::Generic,double> fa_flux0 = 
      regions[1].get_field_accessor(*it).typeify<double>();
    RegionAccessor<AccessorType::Generic,double> fa_q2grp0 = 
      regions[2].get_field_accessor(*it).typeify<double>();
    RegionAccessor<AccessorType::Generic,MomentQuad> fa_qtot = 
      regions[3].get_field_accessor(*it).typeify<MomentQuad>(); 

    ByteOffset sxs_offsets[3], flux0_offsets[3], 
               q2grp0_offsets[3], qtot_offsets[3];
    MomentQuad *sxs_ptr = fa_sxs.raw_rect_ptr<3>(sxs_offsets);
    double *flux0_ptr = fa_flux0.raw_rect_ptr<3>(flux0_offsets);
    double *q2grp0_ptr = fa_q2grp0.raw_rect_ptr<3>(q2grp0_offsets);
    MomentQuad *qtot_ptr = fa_qtot.raw_rect_ptr<3>(qtot_offsets);
    if (multi_moment) {
      RegionAccessor<AccessorType::Generic,MomentTriple> fa_fluxm = 
        regions[4].get_field_accessor(*it).typeify<MomentTriple>();
      RegionAccessor<AccessorType::Generic,MomentTriple> fa_q2grpm = 
        regions[5].get_field_accessor(*it).typeify<MomentTriple>();

      ByteOffset fluxm_offsets[3], q2grpm_offsets[3];
      MomentTriple *fluxm_ptr = fa_fluxm.raw_rect_ptr<3>(fluxm_offsets);
      MomentTriple *q2grpm_ptr = fa_q2grpm.raw_rect_ptr<3>(q2grpm_offsets);
      for (int z = subgrid_bounds.lo[2]; z <= subgrid_bounds.hi[2]; z++) {
        for (int y = subgrid_bounds.lo[1]; y <= subgrid_bounds.hi[1]; y++) {
          for (int x = subgrid_bounds.lo[0]; x <= subgrid_bounds.hi[0]; x++) {
            MomentQuad sxs_quad = *(sxs_ptr + x * sxs_offsets[0] + 
                y * sxs_offsets[1] + z * sxs_offsets[2]);
            const double q0 = *(q2grp0_ptr + x * q2grp0_offsets[0] +
                y * q2grp0_offsets[1] + z * q2grp0_offsets[2]);
            const double flux0 = *(flux0_ptr + x * flux0_offsets[0] + 
                y * flux0_offsets[1] + z * flux0_offsets[2]);
            MomentQuad quad;
            quad[0] = q0 + flux0 * sxs_quad[0]; 
            MomentTriple qom = *(q2grpm_ptr + x * q2grpm_offsets[0] + 
                y * q2grpm_offsets[1] + z * q2grpm_offsets[2]);
            MomentTriple fluxm = *(fluxm_ptr + x * fluxm_offsets[0] +
                y * fluxm_offsets[1] + z * fluxm_offsets[2]);
            int moment = 0;
            for (int l = 1; l < Snap::num_moments; l++) {
              for (int i = 0; i < Snap::lma[l]; i++)
                quad[moment+i+1] = qom[moment+i] + fluxm[moment+i] * sxs_quad[l];
              moment += Snap::lma[l];
            }
            *(qtot_ptr + x * qtot_offsets[0] + y * qtot_offsets[1] + 
                z * qtot_offsets[2]) = quad;
          }
        }
      }
    } else {
      for (int z = subgrid_bounds.lo[2]; z <= subgrid_bounds.hi[2]; z++) {
        for (int y = subgrid_bounds.lo[1]; y <= subgrid_bounds.hi[1]; y++) {
          for (int x = subgrid_bounds.lo[0]; x <= subgrid_bounds.hi[0]; x++) {
            MomentQuad sxs_quad = *(sxs_ptr + x * sxs_offsets[0] + 
                y * sxs_offsets[1] + z * sxs_offsets[2]);
            const double q0 = *(q2grp0_ptr + x * q2grp0_offsets[0] +
                y * q2grp0_offsets[1] + z * q2grp0_offsets[2]);
            const double flux0 = *(flux0_ptr + x * flux0_offsets[0] + 
                y * flux0_offsets[1] + z * flux0_offsets[2]);
            MomentQuad quad;
            quad[0] = q0 + flux0 * sxs_quad[0];
            *(qtot_ptr + x * qtot_offsets[0] + y * qtot_offsets[1] + 
                z * qtot_offsets[2]) = quad;
          }
        }
      }
    }
  }
#endif
}

#ifdef USE_GPU_KERNELS
  extern void run_inner_source_single_moment(Rect<3>           subgrid_bounds,
                                             const MomentQuad  *sxs_ptr,
                                             const double      *flux0_ptr,
                                             const double      *q2grp0_ptr,
                                                   MomentQuad  *qtot_ptr,
                                             ByteOffset        sxs_offsets[3],
                                             ByteOffset        flux0_offsets[3],
                                             ByteOffset        q2grp0_offsets[3],
                                             ByteOffset        qtot_offsets[3]);
  extern void run_inner_source_multi_moment(Rect<3> subgrid_bounds,
                                            const MomentQuad   *sxs_ptr,
                                            const double       *flux0_ptr,
                                            const double       *q2grp0_ptr,
                                            const MomentTriple *fluxm_ptr,
                                            const MomentTriple *q2grpm_ptr,
                                                  MomentQuad   *qtot_ptr,
                                            ByteOffset         sxs_offsets[3],
                                            ByteOffset         flux0_offsets[3],
                                            ByteOffset         q2grp0_offsets[3],
                                            ByteOffset         fluxm_offsets[3],
                                            ByteOffset         q2grpm_offsets[3],
                                            ByteOffset         qtot_offsets[3],
                                            const int num_moments, const int lma[4]);
#endif

//------------------------------------------------------------------------------
/*static*/ void CalcInnerSource::gpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
#ifdef USE_GPU_KERNELS
  Domain dom = runtime->get_index_space_domain(ctx, 
          task->regions[0].region.get_index_space());
  Rect<3> subgrid_bounds = dom.get_rect<3>();
  const bool multi_moment = (Snap::num_moments > 1);
  const unsigned num_groups = task->regions[0].privilege_fields.size();
  assert(num_groups == task->regions[1].privilege_fields.size());
  assert(num_groups == task->regions[2].privilege_fields.size());
  for (std::set<FieldID>::const_iterator it = 
        task->regions[0].privilege_fields.begin(); it !=
        task->regions[0].privilege_fields.end(); it++)
  {
    RegionAccessor<AccessorType::Generic,MomentQuad> fa_sxs = 
      regions[0].get_field_accessor(*it).typeify<MomentQuad>();
    RegionAccessor<AccessorType::Generic,double> fa_flux0 = 
      regions[1].get_field_accessor(*it).typeify<double>();
    RegionAccessor<AccessorType::Generic,double> fa_q2grp0 = 
      regions[2].get_field_accessor(*it).typeify<double>();
    RegionAccessor<AccessorType::Generic,MomentQuad> fa_qtot = 
      regions[3].get_field_accessor(*it).typeify<MomentQuad>(); 

    ByteOffset sxs_offsets[3], flux0_offsets[3], 
               q2grp0_offsets[3], qtot_offsets[3];
    MomentQuad *sxs_ptr = fa_sxs.raw_rect_ptr<3>(sxs_offsets);
    double *flux0_ptr = fa_flux0.raw_rect_ptr<3>(flux0_offsets);
    double *q2grp0_ptr = fa_q2grp0.raw_rect_ptr<3>(q2grp0_offsets);
    MomentQuad *qtot_ptr = fa_qtot.raw_rect_ptr<3>(qtot_offsets);
    if (multi_moment) {
      RegionAccessor<AccessorType::Generic,MomentTriple> fa_fluxm = 
        regions[4].get_field_accessor(*it).typeify<MomentTriple>();
      RegionAccessor<AccessorType::Generic,MomentTriple> fa_q2grpm = 
        regions[5].get_field_accessor(*it).typeify<MomentTriple>();

      ByteOffset fluxm_offsets[3], q2grpm_offsets[3];
      MomentTriple *fluxm_ptr = fa_fluxm.raw_rect_ptr<3>(fluxm_offsets);
      MomentTriple *q2grpm_ptr = fa_q2grpm.raw_rect_ptr<3>(q2grpm_offsets);
      run_inner_source_multi_moment(subgrid_bounds, sxs_ptr, flux0_ptr, q2grp0_ptr,
                                    fluxm_ptr, q2grpm_ptr, qtot_ptr, sxs_offsets,
                                    flux0_offsets, q2grp0_offsets, fluxm_offsets,
                                    q2grpm_offsets, qtot_offsets, 
                                    Snap::num_moments, Snap::lma);
    } else {
      run_inner_source_single_moment(subgrid_bounds, sxs_ptr, flux0_ptr, q2grp0_ptr,
                                     qtot_ptr, sxs_offsets, flux0_offsets, 
                                     q2grp0_offsets, qtot_offsets);
    }
  }
#else
  assert(false);
#endif
#endif
}

//------------------------------------------------------------------------------
TestInnerConvergence::TestInnerConvergence(const Snap &snap, 
                                           const Predicate &pred,
                                           const SnapArray &flux0,
                                           const SnapArray &flux0pi,
                                           const Future &true_future)
  : SnapTask<TestInnerConvergence, Snap::TEST_INNER_CONVERGENCE_TASK_ID>(
      snap, snap.get_launch_bounds(), pred)
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
  register_cpu_variant<bool, fast_implementation>(true/*leaf*/);
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
#ifndef NO_COMPUTE
  log_snap.info("Running Test Inner Convergence");

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
#else
  return false;
#endif
}

//------------------------------------------------------------------------------
/*static*/ bool TestInnerConvergence::fast_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  log_snap.info("Running Fast Test Inner Convergence");

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

    ByteOffset flux0_offsets[3], flux0pi_offsets[3];

    double *flux0_ptr = fa_flux0.raw_rect_ptr<3>(flux0_offsets);
    double *flux0pi_ptr = fa_flux0pi.raw_rect_ptr<3>(flux0pi_offsets);
    for (int z = subgrid_bounds.lo[2]; z <= subgrid_bounds.hi[2]; z++) {
      for (int y = subgrid_bounds.lo[1]; y <= subgrid_bounds.hi[1]; y++) {
        for (int x = subgrid_bounds.lo[0]; x <= subgrid_bounds.hi[0]; x++) {
          double flux0pi = *(flux0pi_ptr + x * flux0pi_offsets[0] + 
              y * flux0pi_offsets[1] + z * flux0pi_offsets[2]);
          double df = 1.0;
          if (fabs(flux0pi) < tolr) {
            flux0pi = 1.0;
            df = 0.0;
          }
          double flux0 = *(flux0_ptr + x * flux0_offsets[0] + 
              y * flux0_offsets[1] + z * flux0_offsets[2]);
          df = fabs( (flux0 / flux0pi) - df );
          if (df > epsi)
            return false;
        }
      }
    }
  }
  return true;
#else
  return false;
#endif
}

#ifdef USE_GPU_KERNELS
extern bool run_inner_convergence(Rect<3> subgrid_bounds,
                                  std::vector<double*> flux0_ptrs,
                                  std::vector<double*> flux0pi_ptrs,
                                  ByteOffset flux0_offsets[3], 
                                  ByteOffset flux0pi_offsets[3],
                                  const double epsi);
#endif

//------------------------------------------------------------------------------
/*static*/ bool TestInnerConvergence::gpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  log_snap.info("Running GPU Test Inner Convergence");
#ifdef USE_GPU_KERNELS
  std::vector<double*> flux0_ptrs(task->regions[0].privilege_fields.size());
  std::vector<double*> flux0pi_ptrs(flux0_ptrs.size());
  ByteOffset flux0_offsets[3], flux0pi_offsets[3];
  unsigned idx = 0;
  for (std::set<FieldID>::const_iterator it = 
        task->regions[0].privilege_fields.begin(); it !=
        task->regions[0].privilege_fields.end(); it++, idx++)
  {
    RegionAccessor<AccessorType::Generic,double> fa_flux0 = 
      regions[0].get_field_accessor(*it).typeify<double>();
    RegionAccessor<AccessorType::Generic,double> fa_flux0pi = 
      regions[1].get_field_accessor(*it).typeify<double>();
    if (idx == 0) {
      flux0_ptrs[idx] = fa_flux0.raw_rect_ptr<3>(flux0_offsets);
      flux0pi_ptrs[idx] = fa_flux0pi.raw_rect_ptr<3>(flux0pi_offsets);
    } else {
      ByteOffset temp_offsets[3];
      flux0_ptrs[idx] = fa_flux0.raw_rect_ptr<3>(temp_offsets);
      assert(offsets_match<3>(temp_offsets, flux0_offsets));
      flux0pi_ptrs[idx] = fa_flux0pi.raw_rect_ptr<3>(temp_offsets);
      assert(offsets_match<3>(temp_offsets, flux0pi_offsets));
    }
  }

  Domain dom = runtime->get_index_space_domain(ctx, 
          task->regions[0].region.get_index_space());
  Rect<3> subgrid_bounds = dom.get_rect<3>();
  const double epsi = Snap::convergence_eps;
  return run_inner_convergence(subgrid_bounds, flux0_ptrs, flux0pi_ptrs,
                               flux0_offsets, flux0pi_offsets, epsi);
#else
  assert(false);
#endif
#endif
  return false;
}

