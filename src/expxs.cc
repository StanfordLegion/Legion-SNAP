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

#include "snap.h"
#include "expxs.h"
#include "legion_stl.h"

extern Legion::Logger log_snap;

using namespace Legion::STL;

//------------------------------------------------------------------------------
ExpandCrossSection::ExpandCrossSection(const Snap &snap,const SnapArray<1> &sig,
                           const SnapArray<3> &mat, const SnapArray<3> &xs, 
                           int start, int stop)
  : SnapTask<ExpandCrossSection, Snap::EXPAND_CROSS_SECTION_TASK_ID>(
      snap, snap.get_launch_bounds(), Predicate::TRUE_PRED), 
    group_start(start), group_stop(stop)
//------------------------------------------------------------------------------
{
  global_arg = TaskArgument(&group_start, 2*sizeof(group_start));
  if (group_start == group_stop) {
    sig.add_region_requirement(READ_ONLY, *this, 
                               SNAP_ENERGY_GROUP_FIELD(group_start));
    mat.add_projection_requirement(READ_ONLY, *this, Snap::FID_SINGLE);
    xs.add_projection_requirement(WRITE_DISCARD, *this, 
                                  SNAP_ENERGY_GROUP_FIELD(group_start));
  } else {
    std::vector<Snap::SnapFieldID> fields((group_stop - group_start) + 1);
    for (int group = group_start; group <= group_stop; group++)
      fields[group-group_start] = SNAP_ENERGY_GROUP_FIELD(group);
    sig.add_region_requirement(READ_ONLY, *this, fields);
    mat.add_projection_requirement(READ_ONLY, *this, Snap::FID_SINGLE);
    xs.add_projection_requirement(WRITE_DISCARD, *this, fields);
  }
}

//------------------------------------------------------------------------------
/*static*/ void ExpandCrossSection::preregister_cpu_variants(void)
//------------------------------------------------------------------------------
{
  ExecutionConstraintSet execution_constraints;
  // Need x86 CPU
  execution_constraints.add_constraint(ISAConstraint(X86_ISA));
  TaskLayoutConstraintSet layout_constraints;
  // All regions need to be SOA
  for (unsigned idx = 0; idx < 3; idx++)
    layout_constraints.add_layout_constraint(idx/*index*/,
                                             Snap::get_soa_layout());
  register_cpu_variant<cpu_implementation>(execution_constraints,
                                           layout_constraints,
                                           true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ void ExpandCrossSection::preregister_gpu_variants(void)
//------------------------------------------------------------------------------
{
  ExecutionConstraintSet execution_constraints;
  // Need a CUDA GPU 
  execution_constraints.add_constraint(ISAConstraint(CUDA_ISA));
  TaskLayoutConstraintSet layout_constraints;
  // All regions need to be SOA
  for (unsigned idx = 0; idx < 3; idx++)
    layout_constraints.add_layout_constraint(idx/*index*/,
                                             Snap::get_soa_layout());
  register_gpu_variant<
    raw_rect_task_wrapper<double,1,int,3,double,3,gpu_implementation> >(
        execution_constraints, layout_constraints, true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ void ExpandCrossSection::cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  log_snap.info("Running Expand Cross Section");

  const int group_start = *((int*)task->args);
  const int group_stop  = *(((int*)task->args) + 1);
  const int num_groups = (group_stop - group_start) + 1;

  std::vector<Accessor<double,1> > fa_sig(num_groups);
  for (int group = group_start; group <= group_stop; group++)
    fa_sig[group - group_start] = 
      Accessor<double,1>(regions[0], SNAP_ENERGY_GROUP_FIELD(group));
  Accessor<int,3> fa_mat(regions[1], Snap::FID_SINGLE);
  std::vector<Accessor<double,3> > fa_xs(num_groups);
  for (int group = group_start; group <= group_stop; group++)
    fa_xs[group - group_start] = 
      Accessor<double,3>(regions[2], SNAP_ENERGY_GROUP_FIELD(group)); 

  Domain<3> dom = runtime->get_index_space_domain(ctx, 
          IndexSpace<3>(task->regions[2].region.get_index_space()));
  for (DomainIterator<3> itr(dom); itr(); itr++)
  {
    int mat = fa_mat[itr];
    for (int idx = 0; idx < num_groups; idx++)
      fa_xs[idx][itr] = fa_sig[idx][mat];
  }
#endif
}

#ifdef USE_GPU_KERNELS
extern void run_expand_cross_section(const std::vector<double*> &sig_ptrs,
                                     const int *mat_ptr,
                                     const std::vector<double*> &xs_ptrs,
                                     const ByteOffset sig_offsets[1],
                                     const ByteOffset mat_offsets[3],
                                     const ByteOffset xs_offsets[3],
                                     const Rect<3> &subgrid_bounds);
#endif

//------------------------------------------------------------------------------
/*static*/ void ExpandCrossSection::gpu_implementation(
    const Task *task, Context ctx, Runtime *runtime,
    const std::vector<double*> &sig_ptrs, const ByteOffset sig_offsets[1],
    const std::vector<int*> &mat_ptrs, const ByteOffset mat_offsets[3],
    const std::vector<double*> &xs_ptrs, const ByteOffset xs_offsets[3])
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
#ifdef USE_GPU_KERNELS
  log_snap.info("Running GPU Expand Cross Section");

  Domain dom = runtime->get_index_space_domain(ctx, 
          task->regions[2].region.get_index_space());
  Rect<3> subgrid_bounds = dom.get_rect<3>();

  const int *const mat_ptr = mat_ptrs[0];

  run_expand_cross_section(sig_ptrs, mat_ptr, xs_ptrs,
                           sig_offsets, mat_offsets, xs_offsets, subgrid_bounds);
#else
  assert(false);
#endif
#endif
}

//------------------------------------------------------------------------------
ExpandScatteringCrossSection::ExpandScatteringCrossSection(const Snap &snap, 
  const SnapArray<2> &slgg, const SnapArray<3> &mat, const SnapArray<3> &s_xs, 
  int start, int stop)
  : SnapTask<ExpandScatteringCrossSection, 
             Snap::EXPAND_SCATTERING_CROSS_SECTION_TASK_ID>(
                 snap, snap.get_launch_bounds(), Predicate::TRUE_PRED), 
    group_start(start), group_stop(stop)
//------------------------------------------------------------------------------
{
  global_arg = TaskArgument(&group_start, 2*sizeof(group_start));
  if (group_start == group_stop) {
    slgg.add_region_requirement(READ_ONLY, *this, 
                                SNAP_ENERGY_GROUP_FIELD(group_start));
    mat.add_projection_requirement(READ_ONLY, *this, Snap::FID_SINGLE);
    s_xs.add_projection_requirement(WRITE_DISCARD, *this, 
                                    SNAP_ENERGY_GROUP_FIELD(group_start));
  } else {
    std::vector<Snap::SnapFieldID> fields((group_stop - group_start) + 1);
    for (int group = group_start; group <= group_stop; group++)
      fields[group-group_start] = SNAP_ENERGY_GROUP_FIELD(group);
    slgg.add_region_requirement(READ_ONLY, *this, fields);
    mat.add_projection_requirement(READ_ONLY, *this, Snap::FID_SINGLE);
    s_xs.add_projection_requirement(WRITE_DISCARD, *this, fields);
  }
}

//------------------------------------------------------------------------------
/*static*/ void ExpandScatteringCrossSection::preregister_cpu_variants(void)
//------------------------------------------------------------------------------
{
  ExecutionConstraintSet execution_constraints;
  // Need x86 CPU
  execution_constraints.add_constraint(ISAConstraint(X86_ISA));
  TaskLayoutConstraintSet layout_constraints;
  // All regions need to be SOA
  for (unsigned idx = 0; idx < 3; idx++)
    layout_constraints.add_layout_constraint(idx/*index*/,
                                             Snap::get_soa_layout());
  register_cpu_variant<cpu_implementation>(execution_constraints,
                                           layout_constraints,
                                           true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ void ExpandScatteringCrossSection::preregister_gpu_variants(void)
//------------------------------------------------------------------------------
{
  ExecutionConstraintSet execution_constraints;
  // Need a CUDA GPU 
  execution_constraints.add_constraint(ISAConstraint(CUDA_ISA));
  TaskLayoutConstraintSet layout_constraints;
  // All regions need to be SOA
  for (unsigned idx = 0; idx < 3; idx++)
    layout_constraints.add_layout_constraint(idx/*index*/, 
                                             Snap::get_soa_layout());
  register_gpu_variant<
    raw_rect_task_wrapper<MomentQuad,2,int,3,MomentQuad,3,gpu_implementation> >(
        execution_constraints, layout_constraints, true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ void ExpandScatteringCrossSection::cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  log_snap.info("Running Expand Scattering Cross Section");

  const int group_start = *((int*)task->args);
  const int group_stop  = *(((int*)task->args) + 1);
  const int num_groups = (group_stop - group_start) + 1;

  std::vector<Accessor<MomentQuad,2> > fa_slgg(num_groups);
  for (int group = group_start; group <= group_stop; group++)
    fa_slgg[group - group_start] = 
      Accessor<MomentQuad,2>(regions[0], SNAP_ENERGY_GROUP_FIELD(group));
  Accessor<int,3> fa_mat(regions[1], Snap::FID_SINGLE);
  std::vector<Accessor<MomentQuad,3> > fa_xs(num_groups);
  for (int group = group_start; group <= group_stop; group++)
    fa_xs[group - group_start] = 
      Accessor<MomentQuad,3>(regions[2], SNAP_ENERGY_GROUP_FIELD(group));

  Domain<3> dom = runtime->get_index_space_domain(ctx, 
          IndexSpace<3>(task->regions[2].region.get_index_space()));
  for (DomainIterator<3> itr(dom); itr(); itr++)
  {
    int mat = fa_mat[itr];
    for (int idx = 0; idx < num_groups; idx++)
      fa_xs[idx][itr] = fa_slgg[idx][mat][group_start+idx];
  }
#endif
}

#ifdef USE_GPU_KERNELS
extern void run_expand_scattering_cross_section(
                                      const std::vector<MomentQuad*> &slgg_ptrs,
                                      const int *mat_ptr,
                                      const std::vector<MomentQuad*> &xs_ptrs,
                                      const ByteOffset slgg_offsets[3],
                                      const ByteOffset mat_offsets[3],
                                      const ByteOffset xs_offsets[3],
                                      const Rect<3> &subgrid_bounds,
                                      const int group_start);
#endif

//------------------------------------------------------------------------------
/*static*/ void ExpandScatteringCrossSection::gpu_implementation(
    const Task *task, Context ctx, Runtime *runtime,
    const std::vector<MomentQuad*> &slgg_ptrs, const ByteOffset slgg_offsets[2],
    const std::vector<int*> &mat_ptrs, const ByteOffset mat_offsets[3],
    const std::vector<MomentQuad*> &xs_ptrs, const ByteOffset xs_offsets[3])
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
#ifdef USE_GPU_KERNELS
  log_snap.info("Running GPU Expand Scattering Cross Section");

  const int group_start = *((int*)task->args);
  const int *const mat_ptr = mat_ptrs[0];

  Domain dom = runtime->get_index_space_domain(ctx, 
          task->regions[2].region.get_index_space());
  Rect<3> subgrid_bounds = dom.get_rect<3>();

  run_expand_scattering_cross_section(slgg_ptrs, mat_ptr, xs_ptrs,
                                      slgg_offsets, mat_offsets, xs_offsets,
                                      subgrid_bounds, group_start);
#else
  assert(false);
#endif
#endif
}

//------------------------------------------------------------------------------
CalculateGeometryParam::CalculateGeometryParam(const Snap &snap, 
                          const SnapArray<3> &t_xs, const SnapArray<1> &vdelt, 
                          const SnapArray<3> &dinv, int start, int stop)
  : SnapTask<CalculateGeometryParam,
             Snap::CALCULATE_GEOMETRY_PARAM_TASK_ID>(
                 snap, snap.get_launch_bounds(), Predicate::TRUE_PRED), 
             group_start(start), group_stop(stop)
//------------------------------------------------------------------------------
{
  global_arg = TaskArgument(&group_start, 2*sizeof(group_start));
  if (group_start == group_stop) {
    t_xs.add_projection_requirement(READ_ONLY, *this, 
                                    SNAP_ENERGY_GROUP_FIELD(group_start));
    vdelt.add_region_requirement(READ_ONLY, *this, 
                                 SNAP_ENERGY_GROUP_FIELD(group_start));
    dinv.add_projection_requirement(WRITE_DISCARD, *this, 
                                    SNAP_ENERGY_GROUP_FIELD(group_start));
  } else {
    std::vector<Snap::SnapFieldID> fields((group_stop - group_start) + 1);
    for (int group = group_start; group <= group_stop; group++)
      fields[group-group_start] = SNAP_ENERGY_GROUP_FIELD(group);
    t_xs.add_projection_requirement(READ_ONLY, *this, fields); 
    vdelt.add_region_requirement(READ_ONLY, *this, fields); 
    dinv.add_projection_requirement(WRITE_DISCARD, *this, fields);
  }
}

//------------------------------------------------------------------------------
/*static*/ void CalculateGeometryParam::preregister_cpu_variants(void)
//------------------------------------------------------------------------------
{
  ExecutionConstraintSet execution_constraints;
  // Need x86 CPU
  execution_constraints.add_constraint(ISAConstraint(X86_ISA));
  TaskLayoutConstraintSet layout_constraints;
  // All regions need to be SOA
  for (unsigned idx = 0; idx < 3; idx++)
    layout_constraints.add_layout_constraint(idx/*index*/,
                                             Snap::get_soa_layout());
  register_cpu_variant<cpu_implementation>(execution_constraints,
                                           layout_constraints,
                                           true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ void CalculateGeometryParam::preregister_gpu_variants(void)
//------------------------------------------------------------------------------
{
  ExecutionConstraintSet execution_constraints;
  // Need a CUDA GPU with at least sm30
  execution_constraints.add_constraint(ISAConstraint(CUDA_ISA));
  TaskLayoutConstraintSet layout_constraints;
  // All regions need to be SOA
  for (unsigned idx = 0; idx < 3; idx++)
    layout_constraints.add_layout_constraint(idx/*index*/, 
                                             Snap::get_soa_layout());
  register_gpu_variant<
    raw_rect_task_wrapper<double,3,double,1,double,3,gpu_implementation> >(
        execution_constraints, layout_constraints, true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ void CalculateGeometryParam::cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  log_snap.info("Running Calculate Geometry Param");

  const int group_start = *((int*)task->args);
  const int group_stop  = *(((int*)task->args) + 1);

  Domain<3> dom = runtime->get_index_space_domain(ctx, 
          IndexSpace<3>(task->regions[2].region.get_index_space()));

  const size_t buffer_size = Snap::num_angles * sizeof(double);
  double *temp = (double*)malloc(buffer_size);

  for (int group = group_start; group <= group_stop; group++)
  {
    Accessor<double,3> fa_xs(regions[0], SNAP_ENERGY_GROUP_FIELD(group));
    Accessor<double,3> fa_dinv(regions[2], SNAP_ENERGY_GROUP_FIELD(group));
    const double vdelt = 
      Accessor<double,1>(regions[1], SNAP_ENERGY_GROUP_FIELD(group))[0];
    for (DomainIterator<3> itr(dom); itr(); itr++)
    {
      const double xs = fa_xs[itr];
      for (int ang = 0; ang < Snap::num_angles; ang++)
        temp[ang] = 1.0 / (xs + vdelt + Snap::hi * Snap::mu[ang] + 
                           Snap::hj * Snap::eta[ang] + Snap::hk * Snap::xi[ang]);
      memcpy(fa_dinv.ptr(itr), temp, buffer_size);
    }
  }

  free(temp);
#endif
}

#ifdef USE_GPU_KERNELS
extern void run_geometry_param(const std::vector<double*> &xs_ptrs,
                               const std::vector<double*> &dinv_ptrs,
                               const ByteOffset xs_offsets[3],
                               const ByteOffset dinv_offsets[3],
                               const std::vector<double> &vdelts,
                               const double hi, const double hj, const double hk,
                               const Rect<3> &subgrid_bounds, const int num_angles);
#endif

//------------------------------------------------------------------------------
/*static*/ void CalculateGeometryParam::gpu_implementation(
    const Task *task, Context ctx, Runtime *runtime,
    const std::vector<double*> &xs_ptrs, const ByteOffset xs_offsets[3],
    const std::vector<double*> &vdelt_ptrs, const ByteOffset vdelt_offsets[1],
    const std::vector<double*> &dinv_ptrs, const ByteOffset dinv_offsets[3])
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
#ifdef USE_GPU_KERNELS
  log_snap.info("Running GPU Calculate Geometry Param");

  const int group_start = *((int*)task->args);
  const int group_stop  = *(((int*)task->args) + 1);
  const int num_groups = (group_stop - group_start) + 1;

  Domain dom = runtime->get_index_space_domain(ctx, 
          task->regions[2].region.get_index_space());
  Rect<3> subgrid_bounds = dom.get_rect<3>();

  std::vector<double> vdelts(num_groups);
  for (int idx = 0; idx < num_groups; idx++)
    vdelts[idx] = *(vdelt_ptrs[idx]);

  run_geometry_param(xs_ptrs, dinv_ptrs, xs_offsets, dinv_offsets,
                     vdelts, Snap::hi, Snap::hj, Snap::hk, 
                     subgrid_bounds, Snap::num_angles);
#else
  assert(false);
#endif
#endif
}

