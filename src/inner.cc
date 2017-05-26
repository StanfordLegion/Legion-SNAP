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

#include <cmath>

#include "snap.h"
#include "inner.h"
#include "legion_stl.h"

extern Legion::Logger log_snap;

using namespace Legion::STL;

//------------------------------------------------------------------------------
CalcInnerSource::CalcInnerSource(const Snap &snap, const Predicate &pred,
                       const SnapArray<3> &s_xs, const SnapArray<3> &flux0,
                       const SnapArray<3> &fluxm, const SnapArray<3> &q2grp0,
                       const SnapArray<3> &q2grpm, const SnapArray<3> &qtot)
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
  } else {
    fluxm.add_projection_requirement(NO_ACCESS, *this);
    q2grpm.add_projection_requirement(NO_ACCESS, *this);
  }
}

//------------------------------------------------------------------------------
/*static*/ void CalcInnerSource::preregister_cpu_variants(void)
//------------------------------------------------------------------------------
{
  ExecutionConstraintSet execution_constraints;
  // Need x86 CPU
  execution_constraints.add_constraint(ISAConstraint(X86_ISA));
  TaskLayoutConstraintSet layout_constraints;
  // All regions need to be SOA
  for (unsigned idx = 0; idx < 6; idx++)
    layout_constraints.add_layout_constraint(idx/*index*/,
                                             Snap::get_soa_layout());
  register_cpu_variant<cpu_implementation>(execution_constraints,
                                           layout_constraints,
                                           true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ void CalcInnerSource::preregister_gpu_variants(void)
//------------------------------------------------------------------------------
{
  ExecutionConstraintSet execution_constraints;
  // Need a CUDA GPU with at least sm30
  execution_constraints.add_constraint(ISAConstraint(CUDA_ISA));
  TaskLayoutConstraintSet layout_constraints;
  // All regions need to be SOA
  for (unsigned idx = 0; idx < 6; idx++)
    layout_constraints.add_layout_constraint(idx/*index*/, 
                                             Snap::get_soa_layout());
  register_gpu_variant<
    raw_rect_task_wrapper<MomentQuad,3,double,3,double,3,MomentQuad,3,
                       MomentTriple,3,MomentTriple,3,gpu_implementation> >(
              execution_constraints, layout_constraints, true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ void CalcInnerSource::cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  log_snap.info("Running Calc Inner Source");

  Domain<3> dom = runtime->get_index_space_domain(ctx, 
          IndexSpace<3>(task->regions[0].region.get_index_space()));
  const bool multi_moment = (Snap::num_moments > 1);
  const unsigned num_groups = task->regions[0].privilege_fields.size();
  assert(num_groups == task->regions[1].privilege_fields.size());
  assert(num_groups == task->regions[2].privilege_fields.size());
  for (std::set<FieldID>::const_iterator it = 
        task->regions[0].privilege_fields.begin(); it !=
        task->regions[0].privilege_fields.end(); it++)
  {
    Accessor<MomentQuad,3> fa_sxs(regions[0], *it);
    Accessor<double,3> fa_flux0(regions[1], *it);
    Accessor<double,3> fa_q2grp0(regions[2], *it);
    Accessor<MomentQuad,3> fa_qtot(regions[3], *it);
    if (multi_moment) {
      Accessor<MomentTriple,3> fa_fluxm(regions[4], *it);
      Accessor<MomentTriple,3> fa_q2grpm(regions[5], *it);
      for (DomainIterator<3> itr(dom); itr(); itr++)
      {
        MomentQuad sxs_quad = fa_sxs[itr];
        const double q0 = fa_q2grp0[itr];
        const double flux0 = fa_flux0[itr];
        MomentQuad quad;
        quad[0] = q0 + flux0 * sxs_quad[0];
        MomentTriple qom = fa_q2grpm[itr];
        MomentTriple fm = fa_fluxm[itr];
        int moment = 0;
        for (int l = 1; l < Snap::num_moments; l++) {
          for (int i = 0; i < Snap::lma[l]; i++)
            quad[moment+i+1] = qom[moment+i] + fm[moment+i] * sxs_quad[l];
          moment += Snap::lma[l];
        }
        fa_qtot[itr] = quad;
      }
    } else {
      for (DomainIterator<3> itr(dom); itr(); itr++)
      {
        MomentQuad sxs_quad = fa_sxs[itr];
        const double q0 = fa_q2grp0[itr];
        const double flux0 = fa_flux0[itr];
        MomentQuad quad;
        quad[0] = q0 + flux0 * sxs_quad[0];
        fa_qtot[itr] = quad;
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
                                             const ByteOffset        sxs_offsets[3],
                                             const ByteOffset        flux0_offsets[3],
                                             const ByteOffset        q2grp0_offsets[3],
                                             const ByteOffset        qtot_offsets[3]);
  extern void run_inner_source_multi_moment(Rect<3> subgrid_bounds,
                                            const MomentQuad   *sxs_ptr,
                                            const double       *flux0_ptr,
                                            const double       *q2grp0_ptr,
                                            const MomentTriple *fluxm_ptr,
                                            const MomentTriple *q2grpm_ptr,
                                                  MomentQuad   *qtot_ptr,
                                            const ByteOffset         sxs_offsets[3],
                                            const ByteOffset         flux0_offsets[3],
                                            const ByteOffset         q2grp0_offsets[3],
                                            const ByteOffset         fluxm_offsets[3],
                                            const ByteOffset         q2grpm_offsets[3],
                                            const ByteOffset         qtot_offsets[3],
                                            const int num_moments, const int lma[4]);
#endif

//------------------------------------------------------------------------------
/*static*/ void CalcInnerSource::gpu_implementation(
    const Task *task, Context ctx, Runtime *runtime,
    const std::vector<MomentQuad*> &sxs_ptrs, const ByteOffset sxs_offsets[3],
    const std::vector<double*> &flux0_ptrs, const ByteOffset flux0_offsets[3],
    const std::vector<double*> &q2grp0_ptrs, const ByteOffset q2grp0_offsets[3],
    const std::vector<MomentQuad*> &qtot_ptrs, const ByteOffset qtot_offsets[3],
    const std::vector<MomentTriple*> &fluxm_ptrs, const ByteOffset fluxm_offsets[3],
    const std::vector<MomentTriple*> &q2grpm_ptrs, const ByteOffset q2grpm_offsets[3])
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

  for (int group = 0; group < num_groups; group++)
  {
    const MomentQuad *const sxs_ptr = sxs_ptrs[group];
    const double *const flux0_ptr = flux0_ptrs[group];
    const double *const q2grp0_ptr = q2grp0_ptrs[group];
    MomentQuad *const qtot_ptr = qtot_ptrs[group];
    if (multi_moment) {
      const MomentTriple *const fluxm_ptr = fluxm_ptrs[group];
      const MomentTriple *const q2grpm_ptr = q2grpm_ptrs[group];
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
                                           const SnapArray<3> &flux0,
                                           const SnapArray<3> &flux0pi,
                                           const Future &true_future,
                                           int group_start, int group_stop)
  : SnapTask<TestInnerConvergence, Snap::TEST_INNER_CONVERGENCE_TASK_ID>(
      snap, snap.get_launch_bounds(), pred)
//------------------------------------------------------------------------------
{
  if (group_start == group_stop) {
    // Special case for a single field
    const Snap::SnapFieldID group_field = SNAP_ENERGY_GROUP_FIELD(group_start);
    flux0.add_projection_requirement(READ_ONLY, *this, group_field);
    flux0pi.add_projection_requirement(READ_ONLY, *this, group_field);
  } else {
    // General case for arbitrary set of fields
    std::vector<Snap::SnapFieldID> group_fields((group_stop - group_start) + 1);
    for (int group = group_start; group <= group_stop; group++)
      group_fields[group-group_start] = SNAP_ENERGY_GROUP_FIELD(group);
    flux0.add_projection_requirement(READ_ONLY, *this, group_fields);
    flux0pi.add_projection_requirement(READ_ONLY, *this, group_fields);
  }
  predicate_false_future = true_future;
}

//------------------------------------------------------------------------------
/*static*/ void TestInnerConvergence::preregister_cpu_variants(void)
//------------------------------------------------------------------------------
{
  ExecutionConstraintSet execution_constraints;
  // Need x86 CPU
  execution_constraints.add_constraint(ISAConstraint(X86_ISA));
  TaskLayoutConstraintSet layout_constraints;
  // All regions need to be SOA
  for (unsigned idx = 0; idx < 2; idx++)
    layout_constraints.add_layout_constraint(idx/*index*/,
                                             Snap::get_soa_layout());
  register_cpu_variant<bool, cpu_implementation>(execution_constraints,
                                                 layout_constraints,
                                                 true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ void TestInnerConvergence::preregister_gpu_variants(void)
//------------------------------------------------------------------------------
{
  ExecutionConstraintSet execution_constraints;
  // Need a CUDA GPU with at least sm30
  execution_constraints.add_constraint(ISAConstraint(CUDA_ISA | SM_30_ISA));
  // Need at least 128B of shared memory
  execution_constraints.add_constraint(
      ResourceConstraint(SHARED_MEMORY_SIZE, GE_EK/*>=*/, 128/*B*/));
  TaskLayoutConstraintSet layout_constraints;
  // All regions need to be SOA
  for (unsigned idx = 0; idx < 2; idx++)
    layout_constraints.add_layout_constraint(idx/*index*/, 
                                             Snap::get_soa_layout());
  register_gpu_variant<bool,
    raw_rect_task_wrapper<bool, double, 3, double, 3, gpu_implementation> >(
        execution_constraints, layout_constraints, true/*leaf*/);
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
  Domain<3> dom = runtime->get_index_space_domain(ctx, 
          IndexSpace<3>(task->regions[0].region.get_index_space()));
  const double tolr = 1.0e-12;
  const double epsi = Snap::convergence_eps;
  // Iterate over all the energy groups
  assert(task->regions[0].privilege_fields.size() == 
         task->regions[1].privilege_fields.size());
  for (std::set<FieldID>::const_iterator it = 
        task->regions[0].privilege_fields.begin(); it !=
        task->regions[0].privilege_fields.end(); it++)
  {
    Accessor<double,3> fa_flux0(regions[0], *it);
    Accessor<double,3> fa_flux0pi(regions[1], *it);
    for (DomainIterator<3> itr(dom); itr(); itr++)
    {
      double flux0pi = fa_flux0pi[itr];
      double df = 1.0;
      if (fabs(flux0pi) < tolr) {
        flux0pi = 1.0;
        df = 0.0;
      }
      double flux0 = fa_flux0[itr];
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

#ifdef USE_GPU_KERNELS
extern bool run_inner_convergence(Rect<3> subgrid_bounds,
                                  const std::vector<double*> flux0_ptrs,
                                  const std::vector<double*> flux0pi_ptrs,
                                  const ByteOffset flux0_offsets[3], 
                                  const ByteOffset flux0pi_offsets[3],
                                  const double epsi);
#endif

//------------------------------------------------------------------------------
/*static*/ bool TestInnerConvergence::gpu_implementation(
  const Task *task, Context ctx, Runtime *runtime,
  const std::vector<double*> &flux0_ptrs, const ByteOffset flux0_offsets[3],
  const std::vector<double*> &flux0pi_ptrs, const ByteOffset flux0pi_offsets[3])
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  log_snap.info("Running GPU Test Inner Convergence");
#ifdef USE_GPU_KERNELS
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

