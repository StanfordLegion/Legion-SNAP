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
#include "outer.h"

extern Legion::Logger log_snap;

//------------------------------------------------------------------------------
CalcOuterSource::CalcOuterSource(const Snap &snap, const Predicate &pred,
                         const SnapArray<3> &qi, const SnapArray<2> &slgg,
                         const SnapArray<3> &mat, const SnapArray<3> &q2rgp0, 
                         const SnapArray<3> &q2grpm, 
                         const SnapArray<3> &flux0, const SnapArray<3> &fluxm)
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
  } else {
    fluxm.add_projection_requirement(NO_ACCESS, *this); // fluxm 
    q2grpm.add_projection_requirement(NO_ACCESS, *this); // qom
  }
}

//------------------------------------------------------------------------------
/*static*/ void CalcOuterSource::preregister_cpu_variants(void)
//------------------------------------------------------------------------------
{
  ExecutionConstraintSet execution_constraints;
  // Need x86 CPU
  execution_constraints.add_constraint(ISAConstraint(X86_ISA));
  // Need L1 cache at least 32 KB
  execution_constraints.add_constraint(
      ResourceConstraint(L1_CACHE_SIZE, GE_EK/*>=*/, 32768/*32 KB*/));
  // Need L1 cache with at least 8 way set associativity
  execution_constraints.add_constraint(
      ResourceConstraint(L1_CACHE_ASSOCIATIVITY, GE_EK/*>=*/, 8));
  TaskLayoutConstraintSet layout_constraints;
  // All regions need to be SOA
  for (unsigned idx = 0; idx < 7; idx++)
    layout_constraints.add_layout_constraint(idx/*index*/,
                                             Snap::get_soa_layout()); 
  register_cpu_variant<cpu_implementation>(execution_constraints,
                                           layout_constraints,
                                           true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ void CalcOuterSource::preregister_gpu_variants(void)
//------------------------------------------------------------------------------
{
  ExecutionConstraintSet execution_constraints;
  // Need a CUDA GPU with at least sm30
  execution_constraints.add_constraint(ISAConstraint(CUDA_ISA | SM_30_ISA));
  // Need at least 48 KB of shared memory
  execution_constraints.add_constraint(
      ResourceConstraint(SHARED_MEMORY_SIZE, GE_EK/*>=*/, 49152/*48KB*/));
  // Need at least 64K registers
  execution_constraints.add_constraint(
      ResourceConstraint(REGISTER_FILE_SIZE, GE_EK/*>=*/, 65536/*registers*/));
  // Need at least two CTAs per SM for performance
  execution_constraints.add_constraint(LaunchConstraint(LEGION_CTAS_PER_SM, 2));
  TaskLayoutConstraintSet layout_constraints;
  // All regions need to be SOA
  for (unsigned idx = 0; idx < 7; idx++)
    layout_constraints.add_layout_constraint(idx/*index*/, 
                                             Snap::get_soa_layout());
  register_gpu_variant<gpu_implementation>(execution_constraints,
                                           layout_constraints,
                                           true/*leaf*/);
}

static int gcd(int a, int b)
{
  if (a < b) return gcd(a, b-a);
  if (a > b) return gcd(a-b, b);
  return a;
}

//------------------------------------------------------------------------------
/*static*/ void CalcOuterSource::cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  log_snap.info("Running Calc Outer Source");

  Domain<3> dom = runtime->get_index_space_domain(ctx, 
          IndexSpace<3>(task->regions[0].region.get_index_space()));
  const bool multi_moment = (Snap::num_moments > 1);
  const int num_groups = task->regions[0].privilege_fields.size();
  assert(num_groups == int(task->regions[1].privilege_fields.size()));
  assert(num_groups == int(task->regions[4].privilege_fields.size()));
  // Make the accessors for all the groups up front
  std::vector<AccessorRO<double,3> > fa_qi0(num_groups);
  std::vector<AccessorRO<double,3> > fa_flux0(num_groups);
  std::vector<AccessorRO<MomentQuad,2> > fa_slgg(num_groups);
  std::vector<AccessorWO<double,3> > fa_qo0(num_groups);
  std::vector<AccessorRO<MomentTriple,3> > fa_fluxm(multi_moment ? num_groups:0);
  std::vector<AccessorWO<MomentTriple,3> > fa_qom(multi_moment ? num_groups : 0);
  // Field spaces are all the same so this is safe
  int g = 0;
  for (std::set<FieldID>::const_iterator it = 
        task->regions[0].privilege_fields.begin(); it !=
        task->regions[0].privilege_fields.end(); it++, g++)
  {
    fa_qi0[g] = AccessorRO<double,3>(regions[0], *it);
    fa_flux0[g] = AccessorRO<double,3>(regions[1], *it);
    fa_slgg[g] = AccessorRO<MomentQuad,2>(regions[2], *it);
    fa_qo0[g] = AccessorWO<double,3>(regions[4], *it);
    if (multi_moment)
    {
      fa_fluxm[g] = AccessorRO<MomentTriple,3>(regions[5], *it);
      fa_qom[g] = AccessorWO<MomentTriple,3>(regions[6], *it);
    }
  }
  AccessorRO<int,3> fa_mat(regions[3], Snap::FID_SINGLE);

  // We'll block the innermost dimension to get some cache locality
  // This assumes a worst case 128 energy groups and a 32 KB L1 cache
  const Rect<3> bounds = dom.bounds;
  const int strip_size = gcd(bounds.hi[0] - bounds.lo[0] + 1, 32);
  double *flux_strip = (double*)malloc(num_groups*strip_size*sizeof(double));

  for (int z = bounds.lo[2]; z <= bounds.hi[2]; z++) {
    for (int y = bounds.lo[1]; y <= bounds.hi[1]; y++) {
      for (int x = bounds.lo[0]; x <= bounds.hi[0]; x+=strip_size) {
        // Read in the flux strip first
        for (int g = 0; g < num_groups; g++)
          for (int i = 0; i < strip_size; i++) 
            flux_strip[g * strip_size + i] = fa_flux0[g][x+i][y][z];
        // We've loaded all the strips, now do the math
        for (int g1 = 0; g1 < num_groups; g1++) {
          for (int i = 0; i < strip_size; i++) {
            double qo0 = fa_qi0[g1][x+i][y][z];
            // Have to look up the the two materials separately 
            const int mat = fa_mat[x+i][y][z];
            for (int g2 = 0; g2 < num_groups; g2++) {
              if (g1 == g2)
                continue;
              MomentQuad cs = fa_slgg[g1][mat][g2];
              qo0 += cs[0] * flux_strip[g2 * strip_size + i];
            }
            fa_qo0[g1][x+i][y][z] = qo0;
          }
        }
      }
    }
  }
  free(flux_strip);
  // Handle multi-moment
  if (multi_moment) {
    MomentTriple *fluxm_strip = 
      (MomentTriple*)malloc(num_groups*strip_size*sizeof(MomentTriple));
    for (int z = bounds.lo[2]; z <= bounds.hi[2]; z++) {
      for (int y = bounds.lo[1]; y <= bounds.hi[1]; y++) {
        for (int x = bounds.lo[0]; x <= bounds.hi[0]; x+=strip_size) {
          // Read in the fluxm strip first
          for (int g = 0; g < num_groups; g++)
            for (int i = 0; i < strip_size; i++)
              fluxm_strip[g * strip_size+ i] = fa_fluxm[g][x+i][y][z];
          // We've loaded all the strips, now do the math
          for (int g1 = 0; g1 < num_groups; g1++) {
            for (int i = 0; i < strip_size; i++) {
              const int mat = fa_mat[x+i][y][z];
              MomentTriple qom;
              for (int g2 = 0; g2 < num_groups; g2++) {
                if (g1 == g2)
                  continue;
                int moment = 0;
                MomentTriple csm;
                MomentQuad scat = fa_slgg[g1][mat][g2];
                for (int l = 1; l < Snap::num_moments; l++) {
                  for (int j = 0; j < Snap::lma[l]; j++)
                    csm[moment+j] = scat[l];
                  moment += Snap::lma[l];
                }
                MomentTriple fluxm = fluxm_strip[g2 * strip_size + i]; 
                for (int l = 0; l < (Snap::num_moments-1); l++)
                  qom[l] += csm[l] * fluxm[l];
              }
              fa_qom[g1][x+i][y][z] = qom;
            }
          }
        }
      }
    }
    free(fluxm_strip);
  }
#endif
}

#ifdef USE_GPU_KERNELS
  extern void run_flux0_outer_source(Rect<3> subgrid_bounds,
                            const std::vector<AccessorRO<double,3> > &fa_qi0,
                            const std::vector<AccessorRO<double,3> > &fa_flux0,
                            const std::vector<AccessorRO<MomentQuad,2> > &fa_slgg,
                            const std::vector<AccessorWO<double,3> > &fa_qo0,
                            const AccessorRO<int,3> &fa_mat, const int num_groups);
  extern void run_fluxm_outer_source(Rect<3> subgrid_bounds,
                            const std::vector<AccessorRO<MomentTriple,3> > &fa_fluxm,
                            const std::vector<AccessorRO<MomentQuad,2> > &fa_slgg,
                            const std::vector<AccessorWO<MomentTriple,3> > &fa_qom,
                            const AccessorRO<int,3> &fa_mat, const int num_groups, 
                            const int num_moments, const int lma[4]);
#endif

//------------------------------------------------------------------------------
/*static*/ void CalcOuterSource::gpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  log_snap.info("Running GPU Calc Outer Source");
#ifdef USE_GPU_KERNELS
  Domain<3> dom = runtime->get_index_space_domain(ctx, 
          IndexSpace<3>(task->regions[0].region.get_index_space()));
  const bool multi_moment = (Snap::num_moments > 1);
  const int num_groups = task->regions[0].privilege_fields.size();
  assert(num_groups == int(task->regions[1].privilege_fields.size()));
  assert(num_groups == int(task->regions[4].privilege_fields.size()));
  // Make the accessors for all the groups up front
  std::vector<AccessorRO<double,3> > fa_qi0(num_groups);
  std::vector<AccessorRO<double,3> > fa_flux0(num_groups);
  std::vector<AccessorRO<MomentQuad,2> > fa_slgg(num_groups);
  std::vector<AccessorWO<double,3> > fa_qo0(num_groups);
  std::vector<AccessorRO<MomentTriple,3> > fa_fluxm(multi_moment ? num_groups:0);
  std::vector<AccessorWO<MomentTriple,3> > fa_qom(multi_moment ? num_groups : 0);
  // Field spaces are all the same so this is safe
  int g = 0;
  for (std::set<FieldID>::const_iterator it = 
        task->regions[0].privilege_fields.begin(); it !=
        task->regions[0].privilege_fields.end(); it++, g++)
  {
    fa_qi0[g] = AccessorRO<double,3>(regions[0], *it);
    fa_flux0[g] = AccessorRO<double,3>(regions[1], *it);
    fa_slgg[g] = AccessorRO<MomentQuad,2>(regions[2], *it);
    fa_qo0[g] = AccessorWO<double,3>(regions[4], *it);
    if (multi_moment)
    {
      fa_fluxm[g] = AccessorRO<MomentTriple,3>(regions[5], *it);
      fa_qom[g] = AccessorWO<MomentTriple,3>(regions[6], *it);
    }
  }
  AccessorRO<int,3> fa_mat(regions[3], Snap::FID_SINGLE);

  run_flux0_outer_source(dom.bounds, fa_qi0, fa_flux0, fa_slgg, 
                         fa_qo0, fa_mat, num_groups);

  if (multi_moment) {
    run_fluxm_outer_source(dom.bounds, fa_fluxm, fa_slgg, fa_qom,
                           fa_mat, num_groups, Snap::num_moments, Snap::lma);
  }
#else
  assert(false);
#endif
#endif
}

//------------------------------------------------------------------------------
TestOuterConvergence::TestOuterConvergence(const Snap &snap, 
                                           const Predicate &pred,
                                           const SnapArray<3> &flux0,
                                           const SnapArray<3> &flux0po,
                                           const Future &inner_converged,
                                           const Future &true_future,
                                           int group_start, int group_stop)
  : SnapTask<TestOuterConvergence, Snap::TEST_OUTER_CONVERGENCE_TASK_ID>(
      snap, snap.get_launch_bounds(), pred)
//------------------------------------------------------------------------------
{
  if (group_start == group_stop) {
    // Special case for a single field
    const Snap::SnapFieldID group_field = SNAP_ENERGY_GROUP_FIELD(group_start);
    flux0.add_projection_requirement(READ_ONLY, *this, group_field);
    flux0po.add_projection_requirement(READ_ONLY, *this, group_field);
  } else {
    // General case for arbitrary set of fields
    std::vector<Snap::SnapFieldID> group_fields((group_stop - group_start) + 1);
    for (int group = group_start; group <= group_stop; group++)
      group_fields[group-group_start] = SNAP_ENERGY_GROUP_FIELD(group);
    flux0.add_projection_requirement(READ_ONLY, *this, group_fields);
    flux0po.add_projection_requirement(READ_ONLY, *this, group_fields);
  }
  add_future(inner_converged);
  predicate_false_future = true_future;
}

//------------------------------------------------------------------------------
/*static*/ void TestOuterConvergence::preregister_cpu_variants(void)
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
/*static*/ void TestOuterConvergence::preregister_gpu_variants(void)
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
  register_gpu_variant<DeferredValue<bool>, gpu_implementation>(
                                                 execution_constraints,
                                                 layout_constraints,
                                                 true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ bool TestOuterConvergence::cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  log_snap.info("Running Test Outer Convergence");

  // If the inner loop didn't converge, then we can't either
  assert(!task->futures.empty());
  bool inner_converged = task->futures[0].get_result<bool>();
  if (!inner_converged)
    return false;
  // Get the index space domain for iteration
  assert(task->regions[0].region.get_index_space() == 
         task->regions[1].region.get_index_space());
  Domain<3> dom = runtime->get_index_space_domain(ctx, 
          IndexSpace<3>(task->regions[0].region.get_index_space()));
  const double tolr = 1.0e-12;
  const double epsi = 100.0 * Snap::convergence_eps;
  // Iterate over all the energy groups
  assert(task->regions[0].privilege_fields.size() == 
         task->regions[1].privilege_fields.size());
  for (std::set<FieldID>::const_iterator it = 
        task->regions[0].privilege_fields.begin(); it !=
        task->regions[0].privilege_fields.end(); it++)
  {
    AccessorRO<double,3> fa_flux0(regions[0], *it);
    AccessorRO<double,3> fa_flux0po(regions[1], *it);
    for (DomainIterator<3> itr(dom); itr(); itr++)
    {
      double flux0po = fa_flux0po[*itr];
      double df = 1.0;
      if (fabs(flux0po) < tolr) {
        flux0po = 1.0;
        df = 0.0;
      }
      double flux0 = fa_flux0[*itr];
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

#ifdef USE_GPU_KERNELS
extern void run_outer_convergence(Rect<3> subgrid_bounds,
                                  const DeferredValue<bool> &result,
                                  const std::vector<AccessorRO<double,3> > &fa_flux0,
                                  const std::vector<AccessorRO<double,3> > &fa_flux0po,
                                  const double epsi);
#endif

//------------------------------------------------------------------------------
/*static*/ DeferredValue<bool> 
            TestOuterConvergence::gpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
  log_snap.info("Running GPU Test Outer Convergence");
  DeferredValue<bool> result(false);
#ifndef NO_COMPUTE
#ifdef USE_GPU_KERNELS
  // If the inner loop didn't converge, then we can't either
  assert(!task->futures.empty());
  bool inner_converged = task->futures[0].get_result<bool>();
  if (!inner_converged)
    return result;
  // Get the index space domain for iteration
  assert(task->regions[0].region.get_index_space() == 
         task->regions[1].region.get_index_space());
  Domain<3> dom = runtime->get_index_space_domain(ctx, 
          IndexSpace<3>(task->regions[0].region.get_index_space()));
  const double epsi = 100.0 * Snap::convergence_eps;
  std::vector<AccessorRO<double,3> > fa_flux0(
                          task->regions[0].privilege_fields.size());
  std::vector<AccessorRO<double,3> > fa_flux0po(fa_flux0.size());
  unsigned idx = 0;
  for (std::set<FieldID>::const_iterator it = 
        task->regions[0].privilege_fields.begin(); it !=
        task->regions[0].privilege_fields.end(); it++, idx++)
  {
    fa_flux0[idx]   = AccessorRO<double,3>(regions[0], *it);
    fa_flux0po[idx] = AccessorRO<double,3>(regions[1], *it);
  }
  run_outer_convergence(dom.bounds, result, fa_flux0, fa_flux0po, epsi);
#else
  assert(false);
#endif
#endif
  return result;
}

