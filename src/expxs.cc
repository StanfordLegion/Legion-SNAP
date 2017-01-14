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

extern LegionRuntime::Logger::Category log_snap;

using namespace LegionRuntime::Accessor;

//------------------------------------------------------------------------------
ExpandCrossSection::ExpandCrossSection(const Snap &snap, const SnapArray &sig,
                           const SnapArray &mat, const SnapArray &xs, 
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
  register_cpu_variant<fast_implementation>(execution_constraints,
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
  register_gpu_variant<gpu_implementation>(execution_constraints,
                                           layout_constraints,
                                           true/*leaf*/);
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

  std::vector<RegionAccessor<AccessorType::Generic,double> > fa_sig(num_groups);
  for (int group = group_start; group <= group_stop; group++)
    fa_sig[group - group_start] = regions[0].get_field_accessor(
        SNAP_ENERGY_GROUP_FIELD(group)).typeify<double>();
  RegionAccessor<AccessorType::Generic,int> fa_mat = 
    regions[1].get_field_accessor(Snap::FID_SINGLE).typeify<int>();
  std::vector<RegionAccessor<AccessorType::Generic,double> > fa_xs(num_groups);
  for (int group = group_start; group <= group_stop; group++)
    fa_xs[group - group_start] = regions[2].get_field_accessor(
        SNAP_ENERGY_GROUP_FIELD(group)).typeify<double>();

  Domain dom = runtime->get_index_space_domain(ctx, 
          task->regions[2].region.get_index_space());
  Rect<3> subgrid_bounds = dom.get_rect<3>();

  for (GenericPointInRectIterator<3> itr(subgrid_bounds); itr; itr++)
  {
    const DomainPoint dp = DomainPoint::from_point<3>(itr.p);
    int mat = fa_mat.read(dp);
    const DomainPoint indirect = DomainPoint::from_point<1>(Point<1>(mat));
    for (int idx = 0; idx < num_groups; idx++)
      fa_xs[idx].write(dp, fa_sig[idx].read(indirect));
  }
#endif
}

//------------------------------------------------------------------------------
/*static*/ void ExpandCrossSection::fast_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  log_snap.info("Running Fast Expand Cross Section");

  const int group_start = *((int*)task->args);
  const int group_stop  = *(((int*)task->args) + 1);
  const int num_groups = (group_stop - group_start) + 1;

  ByteOffset sig_offsets[1];
  std::vector<double*> sig_ptrs(num_groups);
  for (int group = group_start; group <= group_stop; group++) {
    RegionAccessor<AccessorType::Generic,double> fa_sig = 
      regions[0].get_field_accessor(SNAP_ENERGY_GROUP_FIELD(group)).typeify<double>();
    if (group == group_start) {
      sig_ptrs[0] = fa_sig.raw_rect_ptr<1>(sig_offsets);
    } else {
      ByteOffset temp_offsets[1];
      sig_ptrs[group - group_start] = fa_sig.raw_rect_ptr<1>(temp_offsets);
      assert(offsets_match<1>(temp_offsets, sig_offsets));
    }
  }

  RegionAccessor<AccessorType::Generic,int> fa_mat = 
    regions[1].get_field_accessor(Snap::FID_SINGLE).typeify<int>();
  ByteOffset mat_offsets[3];
  int *mat_ptr = fa_mat.raw_rect_ptr<3>(mat_offsets);

  ByteOffset xs_offsets[3];
  std::vector<double*> xs_ptrs(num_groups);
  for (int group = group_start; group <= group_stop; group++) {
    RegionAccessor<AccessorType::Generic,double> fa_xs = 
      regions[2].get_field_accessor(SNAP_ENERGY_GROUP_FIELD(group)).typeify<double>();
    if (group == group_start) {
      xs_ptrs[0] = fa_xs.raw_rect_ptr<3>(xs_offsets);
    } else {
      ByteOffset temp_offsets[3];
      xs_ptrs[group - group_start] = fa_xs.raw_rect_ptr<3>(temp_offsets);
      assert(offsets_match<3>(temp_offsets, xs_offsets));
    }
  }

  Domain dom = runtime->get_index_space_domain(ctx, 
          task->regions[2].region.get_index_space());
  Rect<3> subgrid_bounds = dom.get_rect<3>();

  for (int z = subgrid_bounds.lo[2]; z <= subgrid_bounds.hi[2]; z++) {
    for (int y = subgrid_bounds.lo[1]; y <= subgrid_bounds.hi[1]; y++) {
      for (int x = subgrid_bounds.lo[0]; x <= subgrid_bounds.hi[0]; x++) {
        const int mat = *(mat_ptr + x * mat_offsets[0] + y * mat_offsets[1] + 
                          z * mat_offsets[2]);
        const ByteOffset offset = mat * sig_offsets[0];
        for (int idx = 0; idx < num_groups; idx++) {
          double val = *(sig_ptrs[idx] + offset);
          *(xs_ptrs[idx] + x * xs_offsets[0] + y * xs_offsets[1] + 
              z * xs_offsets[2]) = val;
        }
      }
    }
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
/*static*/ void ExpandCrossSection::gpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
#ifdef USE_GPU_KERNELS
  log_snap.info("Running GPU Expand Cross Section");

  const int group_start = *((int*)task->args);
  const int group_stop  = *(((int*)task->args) + 1);
  const int num_groups = (group_stop - group_start) + 1;

  ByteOffset sig_offsets[1];
  std::vector<double*> sig_ptrs(num_groups);
  for (int group = group_start; group <= group_stop; group++) {
    RegionAccessor<AccessorType::Generic,double> fa_sig = 
      regions[0].get_field_accessor(SNAP_ENERGY_GROUP_FIELD(group)).typeify<double>();
    if (group == group_start) {
      sig_ptrs[0] = fa_sig.raw_rect_ptr<1>(sig_offsets);
    } else {
      ByteOffset temp_offsets[1];
      sig_ptrs[group - group_start] = fa_sig.raw_rect_ptr<1>(temp_offsets);
      assert(offsets_match<1>(temp_offsets, sig_offsets));
    }
  }

  RegionAccessor<AccessorType::Generic,int> fa_mat = 
    regions[1].get_field_accessor(Snap::FID_SINGLE).typeify<int>();
  ByteOffset mat_offsets[3];
  int *mat_ptr = fa_mat.raw_rect_ptr<3>(mat_offsets);

  ByteOffset xs_offsets[3];
  std::vector<double*> xs_ptrs(num_groups);
  for (int group = group_start; group <= group_stop; group++) {
    RegionAccessor<AccessorType::Generic,double> fa_xs = 
      regions[2].get_field_accessor(SNAP_ENERGY_GROUP_FIELD(group)).typeify<double>();
    if (group == group_start) {
      xs_ptrs[0] = fa_xs.raw_rect_ptr<3>(xs_offsets);
    } else {
      ByteOffset temp_offsets[3];
      xs_ptrs[group - group_start] = fa_xs.raw_rect_ptr<3>(temp_offsets);
      assert(offsets_match<3>(temp_offsets, xs_offsets));
    }
  }

  Domain dom = runtime->get_index_space_domain(ctx, 
          task->regions[2].region.get_index_space());
  Rect<3> subgrid_bounds = dom.get_rect<3>();

  run_expand_cross_section(sig_ptrs, mat_ptr, xs_ptrs,
                           sig_offsets, mat_offsets, xs_offsets, subgrid_bounds);
#else
  assert(false);
#endif
#endif
}

//------------------------------------------------------------------------------
ExpandScatteringCrossSection::ExpandScatteringCrossSection(const Snap &snap, 
  const SnapArray &slgg, const SnapArray &mat, const SnapArray &s_xs, 
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
  register_cpu_variant<fast_implementation>(execution_constraints,
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
  register_gpu_variant<gpu_implementation>(execution_constraints,
                                           layout_constraints,
                                           true/*leaf*/);
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

  std::vector<RegionAccessor<AccessorType::Generic,MomentQuad> > fa_slgg(num_groups);
  for (int group = group_start; group <= group_stop; group++)
    fa_slgg[group - group_start] = regions[0].get_field_accessor(
        SNAP_ENERGY_GROUP_FIELD(group)).typeify<MomentQuad>();
  RegionAccessor<AccessorType::Generic,int> fa_mat = 
    regions[1].get_field_accessor(Snap::FID_SINGLE).typeify<int>();
  std::vector<RegionAccessor<AccessorType::Generic,MomentQuad> > fa_xs(num_groups);
  for (int group = group_start; group <= group_stop; group++)
    fa_xs[group - group_start] = regions[2].get_field_accessor(
        SNAP_ENERGY_GROUP_FIELD(group)).typeify<MomentQuad>();

  Domain dom = runtime->get_index_space_domain(ctx, 
          task->regions[2].region.get_index_space());
  Rect<3> subgrid_bounds = dom.get_rect<3>();

  for (GenericPointInRectIterator<3> itr(subgrid_bounds); itr; itr++)
  {
    const DomainPoint dp = DomainPoint::from_point<3>(itr.p);
    int mat = fa_mat.read(dp);
    for (int idx = 0; idx < num_groups; idx++)
    {
      int point[2] = { mat, group_start + idx };
      const DomainPoint indirect = DomainPoint::from_point<2>(Point<2>(point));
      fa_xs[idx].write(dp, fa_slgg[idx].read(indirect));
    }
  }
#endif
}

//------------------------------------------------------------------------------
/*static*/ void ExpandScatteringCrossSection::fast_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  log_snap.info("Running Fast Expand Scattering Cross Section");

  const int group_start = *((int*)task->args);
  const int group_stop  = *(((int*)task->args) + 1);
  const int num_groups = (group_stop - group_start) + 1;

#ifndef LEGION_ISSUE_214_FIX
  Rect<2> slgg_bounds, actual_bounds;
  slgg_bounds.lo.x[0] = 1; slgg_bounds.lo.x[1] = 0;
  slgg_bounds.hi.x[0] = (Snap::material_layout == Snap::HOMOGENEOUS_LAYOUT) ? 1 : 2;
  slgg_bounds.hi.x[1] = Snap::num_groups - 1;
#endif
  ByteOffset slgg_offsets[2];
  std::vector<MomentQuad*> slgg_ptrs(num_groups);
  for (int group = group_start; group <= group_stop; group++) {
    RegionAccessor<AccessorType::Generic,MomentQuad> fa_slgg = 
      regions[0].get_field_accessor(SNAP_ENERGY_GROUP_FIELD(group)).typeify<MomentQuad>();
    if (group == group_start) {
#ifdef LEGION_ISSUE_214_FIX
      slgg_ptrs[0] = fa_slgg.raw_rect_ptr<2>(slgg_offsets);
#else
      slgg_ptrs[0] = fa_slgg.raw_rect_ptr<2>(slgg_bounds, actual_bounds, slgg_offsets);
      assert(slgg_bounds == actual_bounds);
#endif
    } else {
      ByteOffset temp_offsets[2];
#ifdef LEGION_ISSUE_214_FIX
      slgg_ptrs[group - group_start] = fa_slgg.raw_rect_ptr<2>(temp_offsets);
      assert(offsets_match<2>(temp_offsets, slgg_offsets));
#else
      slgg_ptrs[group - group_start] = 
        fa_slgg.raw_rect_ptr<2>(slgg_bounds, actual_bounds, temp_offsets);
      assert(slgg_bounds == actual_bounds);
      assert(offsets_match<2>(temp_offsets, slgg_offsets));
#endif
    }
  }

  RegionAccessor<AccessorType::Generic,int> fa_mat = 
    regions[1].get_field_accessor(Snap::FID_SINGLE).typeify<int>();
  ByteOffset mat_offsets[3];
  int *mat_ptr = fa_mat.raw_rect_ptr<3>(mat_offsets);

  ByteOffset xs_offsets[3];
  std::vector<MomentQuad*> xs_ptrs(num_groups);
  for (int group = group_start; group <= group_stop; group++) {
    RegionAccessor<AccessorType::Generic,MomentQuad> fa_xs = 
      regions[2].get_field_accessor(SNAP_ENERGY_GROUP_FIELD(group)).typeify<MomentQuad>();
    if (group == group_start) {
      xs_ptrs[0] = fa_xs.raw_rect_ptr<3>(xs_offsets);
    } else {
      ByteOffset temp_offsets[3];
      xs_ptrs[group - group_start] = fa_xs.raw_rect_ptr<3>(temp_offsets);
      assert(offsets_match<3>(temp_offsets, xs_offsets));
    }
  }

  Domain dom = runtime->get_index_space_domain(ctx, 
          task->regions[2].region.get_index_space());
  Rect<3> subgrid_bounds = dom.get_rect<3>();

  for (int z = subgrid_bounds.lo[2]; z <= subgrid_bounds.hi[2]; z++) {
    for (int y = subgrid_bounds.lo[1]; y <= subgrid_bounds.hi[1]; y++) {
      for (int x = subgrid_bounds.lo[0]; x <= subgrid_bounds.hi[0]; x++) {
        const int mat = *(mat_ptr + x * mat_offsets[0] + y * mat_offsets[1] + 
                          z * mat_offsets[2]);
        for (int idx = 0; idx < num_groups; idx++) {
#ifdef LEGION_ISSUE_214_FIX
          MomentQuad val = *(slgg_ptrs[idx] + mat * slgg_offsets[0] + 
                              (group_start + idx) * slgg_offsets[1]);
#else
          MomentQuad val = *(slgg_ptrs[idx] + (mat-1) * slgg_offsets[0] + 
                              (group_start + idx) * slgg_offsets[1]);
#endif
          *(xs_ptrs[idx] + x * xs_offsets[0] + y * xs_offsets[1] + 
              z * xs_offsets[2]) = val;
        }
      }
    }
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
/*static*/ void ExpandScatteringCrossSection::gpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
#ifdef USE_GPU_KERNELS
  log_snap.info("Running GPU Expand Scattering Cross Section");

  const int group_start = *((int*)task->args);
  const int group_stop  = *(((int*)task->args) + 1);
  const int num_groups = (group_stop - group_start) + 1;

#ifndef LEGION_ISSUE_214_FIX
  Rect<2> slgg_bounds, actual_bounds;
  slgg_bounds.lo.x[0] = 1; slgg_bounds.lo.x[1] = 0;
  slgg_bounds.hi.x[0] = (Snap::material_layout == Snap::HOMOGENEOUS_LAYOUT) ? 1 : 2;
  slgg_bounds.hi.x[1] = Snap::num_groups - 1;
#endif
  ByteOffset slgg_offsets[2];
  std::vector<MomentQuad*> slgg_ptrs(num_groups);
  for (int group = group_start; group <= group_stop; group++) {
    RegionAccessor<AccessorType::Generic,MomentQuad> fa_slgg = 
      regions[0].get_field_accessor(SNAP_ENERGY_GROUP_FIELD(group)).typeify<MomentQuad>();
    if (group == group_start) {
#ifdef LEGION_ISSUE_214_FIX
      slgg_ptrs[0] = fa_slgg.raw_rect_ptr<2>(slgg_offsets);
#else
      slgg_ptrs[0] = fa_slgg.raw_rect_ptr<2>(slgg_bounds, actual_bounds, slgg_offsets);
      assert(slgg_bounds == actual_bounds);
#endif
    } else {
      ByteOffset temp_offsets[2];
#ifdef LEGION_ISSUE_214_FIX
      slgg_ptrs[group - group_start] = fa_slgg.raw_rect_ptr<2>(temp_offsets);
      assert(offsets_match<2>(temp_offsets, slgg_offsets));
#else
      slgg_ptrs[group - group_start] = 
        fa_slgg.raw_rect_ptr<2>(slgg_bounds, actual_bounds, temp_offsets);
      assert(slgg_bounds == actual_bounds);
      assert(offsets_match<2>(temp_offsets, slgg_offsets));
#endif
    }
  }

  RegionAccessor<AccessorType::Generic,int> fa_mat = 
    regions[1].get_field_accessor(Snap::FID_SINGLE).typeify<int>();
  ByteOffset mat_offsets[3];
  int *mat_ptr = fa_mat.raw_rect_ptr<3>(mat_offsets);

  ByteOffset xs_offsets[3];
  std::vector<MomentQuad*> xs_ptrs(num_groups);
  for (int group = group_start; group <= group_stop; group++) {
    RegionAccessor<AccessorType::Generic,MomentQuad> fa_xs = 
      regions[2].get_field_accessor(SNAP_ENERGY_GROUP_FIELD(group)).typeify<MomentQuad>();
    if (group == group_start) {
      xs_ptrs[0] = fa_xs.raw_rect_ptr<3>(xs_offsets);
    } else {
      ByteOffset temp_offsets[3];
      xs_ptrs[group - group_start] = fa_xs.raw_rect_ptr<3>(temp_offsets);
      assert(offsets_match<3>(temp_offsets, xs_offsets));
    }
  }

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
                                  const SnapArray &t_xs, const SnapArray &vdelt, 
                                  const SnapArray &dinv, int start, int stop)
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
  register_cpu_variant<fast_implementation>(execution_constraints,
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
  register_gpu_variant<gpu_implementation>(execution_constraints,
                                           layout_constraints,
                                           true/*leaf*/);
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

  Domain dom = runtime->get_index_space_domain(ctx, 
          task->regions[2].region.get_index_space());
  Rect<3> subgrid_bounds = dom.get_rect<3>();

  const size_t buffer_size = Snap::num_angles * sizeof(double);
  double *temp = (double*)malloc(buffer_size);

  for (int group = group_start; group <= group_stop; group++)
  {
    RegionAccessor<AccessorType::Generic,double> fa_xs = 
      regions[0].get_field_accessor(
          SNAP_ENERGY_GROUP_FIELD(group)).typeify<double>();
    RegionAccessor<AccessorType::Generic> fa_dinv = 
      regions[2].get_field_accessor(SNAP_ENERGY_GROUP_FIELD(group));
    const double vdelt = regions[1].get_field_accessor(
       SNAP_ENERGY_GROUP_FIELD(group)).typeify<double>().read(
        DomainPoint::from_point<1>(Point<1>::ZEROES()));

    for (GenericPointInRectIterator<3> itr(subgrid_bounds); itr; itr++)
    {
      const DomainPoint dp = DomainPoint::from_point<3>(itr.p);
      const double xs = fa_xs.read(dp);
      for (int ang = 0; ang < Snap::num_angles; ang++)
        temp[ang] = 1.0 / (xs + vdelt + Snap::hi * Snap::mu[ang] + 
                           Snap::hj * Snap::eta[ang] + Snap::hk * Snap::xi[ang]);
      fa_dinv.write_untyped(dp, temp, buffer_size);
    }
  }

  free(temp);
#endif
}

//------------------------------------------------------------------------------
/*static*/ void CalculateGeometryParam::fast_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  log_snap.info("Running Fast Calculate Geometry Param");

  const int group_start = *((int*)task->args);
  const int group_stop  = *(((int*)task->args) + 1);

  Domain dom = runtime->get_index_space_domain(ctx, 
          task->regions[2].region.get_index_space());
  Rect<3> subgrid_bounds = dom.get_rect<3>();

  for (int group = group_start; group <= group_stop; group++)
  {
    RegionAccessor<AccessorType::Generic,double> fa_xs = 
      regions[0].get_field_accessor(
          SNAP_ENERGY_GROUP_FIELD(group)).typeify<double>();
    ByteOffset xs_offsets[3];
    double *xs_ptr = fa_xs.raw_rect_ptr<3>(xs_offsets);

    RegionAccessor<AccessorType::Generic> fa_dinv = 
      regions[2].get_field_accessor(SNAP_ENERGY_GROUP_FIELD(group));
    ByteOffset dinv_offsets[3];
    double *dinv_ptr = (double*)fa_dinv.raw_rect_ptr<3>(dinv_offsets);

    const double vdelt = regions[1].get_field_accessor(
       SNAP_ENERGY_GROUP_FIELD(group)).typeify<double>().read(
        DomainPoint::from_point<1>(Point<1>::ZEROES()));

    for (int z = subgrid_bounds.lo[2]; z <= subgrid_bounds.hi[2]; z++) {
      for (int y = subgrid_bounds.lo[1]; y <= subgrid_bounds.hi[1]; y++) {
        for (int x = subgrid_bounds.lo[0]; x <= subgrid_bounds.hi[0]; x++) {
          const double xs = *(xs_ptr + x * xs_offsets[0] + y * xs_offsets[1] +
                              z * xs_offsets[2]);
          double *out_ptr = dinv_ptr + x * dinv_offsets[0] + y * dinv_offsets[1] +
                              z * dinv_offsets[2];
          for (int ang = 0; ang < Snap::num_angles; ang++) 
            out_ptr[ang] = 1.0 / (xs + vdelt + Snap::hi * Snap::mu[ang] + 
                          Snap::hj * Snap::eta[ang] + Snap::hk * Snap::xi[ang]);
        }
      }
    }
  }
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
/*static*/ void CalculateGeometryParam::gpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
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

  ByteOffset xs_offsets[3];
  std::vector<double*> xs_ptrs(num_groups);
  for (int idx = 0; idx < num_groups; idx++) {
    RegionAccessor<AccessorType::Generic,double> fa_xs = 
      regions[0].get_field_accessor(
          SNAP_ENERGY_GROUP_FIELD(group_start + idx)).typeify<double>();
    if (idx == 0) {
      xs_ptrs[idx] = fa_xs.raw_rect_ptr<3>(xs_offsets);
    } else {
      ByteOffset temp_offsets[3];
      xs_ptrs[idx] = fa_xs.raw_rect_ptr<3>(temp_offsets);
      assert(offsets_match<3>(temp_offsets, xs_offsets));
    }
  }

  std::vector<double> vdelts(num_groups);
  for (int idx = 0; idx < num_groups; idx++)
    vdelts[idx] = regions[1].get_field_accessor(
       SNAP_ENERGY_GROUP_FIELD(group_start + idx)).typeify<double>().read(
        DomainPoint::from_point<1>(Point<1>::ZEROES()));

  ByteOffset dinv_offsets[3];
  std::vector<double*> dinv_ptrs(num_groups);
  for (int idx = 0; idx < num_groups; idx++) {
    RegionAccessor<AccessorType::Generic> fa_dinv = 
      regions[2].get_field_accessor(SNAP_ENERGY_GROUP_FIELD(group_start + idx));
    if (idx == 0) {
      dinv_ptrs[idx] = (double*)fa_dinv.raw_rect_ptr<3>(dinv_offsets);
    } else {
      ByteOffset temp_offsets[3];
      dinv_ptrs[idx] = (double*)fa_dinv.raw_rect_ptr<3>(temp_offsets);
      assert(offsets_match<3>(temp_offsets, dinv_offsets));
    }
  }

  run_geometry_param(xs_ptrs, dinv_ptrs, xs_offsets, dinv_offsets,
                     vdelts, Snap::hi, Snap::hj, Snap::hk, 
                     subgrid_bounds, Snap::num_angles);
#else
  assert(false);
#endif
#endif
}

