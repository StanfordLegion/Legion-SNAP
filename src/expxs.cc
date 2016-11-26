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
#include "expxs.h"

using namespace LegionRuntime::Accessor;

//------------------------------------------------------------------------------
ExpandCrossSection::ExpandCrossSection(const Snap &snap, const SnapArray &sig,
                           const SnapArray &mat, const SnapArray &xs, int g)
  : SnapTask<ExpandCrossSection, Snap::EXPAND_CROSS_SECTION_TASK_ID>(
      snap, snap.get_launch_bounds(), Predicate::TRUE_PRED), group(g)
//------------------------------------------------------------------------------
{
  global_arg = TaskArgument(&group, sizeof(group));
  sig.add_region_requirement(READ_ONLY, *this, SNAP_ENERGY_GROUP_FIELD(group));
  mat.add_projection_requirement(READ_ONLY, *this, Snap::FID_SINGLE);
  xs.add_projection_requirement(WRITE_DISCARD, *this, 
                                SNAP_ENERGY_GROUP_FIELD(group));
}

//------------------------------------------------------------------------------
/*static*/ void ExpandCrossSection::preregister_cpu_variants(void)
//------------------------------------------------------------------------------
{
  register_cpu_variant<cpu_implementation>(true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ void ExpandCrossSection::preregister_gpu_variants(void)
//------------------------------------------------------------------------------
{
  register_gpu_variant<gpu_implementation>(true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ void ExpandCrossSection::cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  const int group = *((int*)task->args);
  RegionAccessor<AccessorType::Generic,double> fa_sig = 
    regions[0].get_field_accessor(
        SNAP_ENERGY_GROUP_FIELD(group)).typeify<double>();
  RegionAccessor<AccessorType::Generic,int> fa_mat = 
    regions[1].get_field_accessor(Snap::FID_SINGLE).typeify<int>();
  RegionAccessor<AccessorType::Generic,double> fa_xs = 
    regions[2].get_field_accessor(
        SNAP_ENERGY_GROUP_FIELD(group)).typeify<double>();

  Domain dom = runtime->get_index_space_domain(ctx, 
          task->regions[2].region.get_index_space());
  Rect<3> subgrid_bounds = dom.get_rect<3>();

  for (GenericPointInRectIterator<3> itr(subgrid_bounds); itr; itr++)
  {
    const DomainPoint dp = DomainPoint::from_point<3>(itr.p);
    int mat = fa_mat.read(dp);
    const DomainPoint indirect = DomainPoint::from_point<1>(Point<1>(mat));
    fa_xs.write(dp, fa_sig.read(indirect));
  }
#endif
}

//------------------------------------------------------------------------------
/*static*/ void ExpandCrossSection::gpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  assert(false);
#endif
}

//------------------------------------------------------------------------------
ExpandScatteringCrossSection::ExpandScatteringCrossSection(const Snap &snap, 
  const SnapArray &slgg, const SnapArray &mat, const SnapArray &s_xs, int g)
  : SnapTask<ExpandScatteringCrossSection, 
             Snap::EXPAND_SCATTERING_CROSS_SECTION_TASK_ID>(
                 snap, snap.get_launch_bounds(), Predicate::TRUE_PRED), group(g)
//------------------------------------------------------------------------------
{
  global_arg = TaskArgument(&group, sizeof(group));
  slgg.add_region_requirement(READ_ONLY, *this, 
                              SNAP_ENERGY_GROUP_FIELD(group));
  mat.add_projection_requirement(READ_ONLY, *this, Snap::FID_SINGLE);
  s_xs.add_projection_requirement(WRITE_DISCARD, *this, 
                                  SNAP_ENERGY_GROUP_FIELD(group));
}

//------------------------------------------------------------------------------
/*static*/ void ExpandScatteringCrossSection::preregister_cpu_variants(void)
//------------------------------------------------------------------------------
{
  register_cpu_variant<cpu_implementation>(true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ void ExpandScatteringCrossSection::preregister_gpu_variants(void)
//------------------------------------------------------------------------------
{
  register_gpu_variant<gpu_implementation>(true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ void ExpandScatteringCrossSection::cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  const int group = *((int*)task->args);
  RegionAccessor<AccessorType::Generic,MomentQuad> fa_slgg = 
    regions[0].get_field_accessor(
        SNAP_ENERGY_GROUP_FIELD(group)).typeify<MomentQuad>();
  RegionAccessor<AccessorType::Generic,int> fa_mat = 
    regions[1].get_field_accessor(Snap::FID_SINGLE).typeify<int>();
  RegionAccessor<AccessorType::Generic,MomentQuad> fa_xs = 
    regions[2].get_field_accessor(
        SNAP_ENERGY_GROUP_FIELD(group)).typeify<MomentQuad>();

  Domain dom = runtime->get_index_space_domain(ctx, 
          task->regions[2].region.get_index_space());
  Rect<3> subgrid_bounds = dom.get_rect<3>();

  for (GenericPointInRectIterator<3> itr(subgrid_bounds); itr; itr++)
  {
    const DomainPoint dp = DomainPoint::from_point<3>(itr.p);
    int mat = fa_mat.read(dp);
    int point[2] = { mat, group };
    const DomainPoint indirect = DomainPoint::from_point<2>(Point<2>(point));
    fa_xs.write(dp, fa_slgg.read(indirect));
  }
#endif
}

//------------------------------------------------------------------------------
/*static*/ void ExpandScatteringCrossSection::gpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  assert(false);
#endif
}

//------------------------------------------------------------------------------
CalculateGeometryParam::CalculateGeometryParam(const Snap &snap, 
                                  const SnapArray &t_xs, const SnapArray &vdelt, 
                                  const SnapArray &dinv, int g)
  : SnapTask<CalculateGeometryParam,
             Snap::CALCULATE_GEOMETRY_PARAM_TASK_ID>(
                 snap, snap.get_launch_bounds(), Predicate::TRUE_PRED), group(g)
//------------------------------------------------------------------------------
{
  global_arg = TaskArgument(&group, sizeof(group));
  t_xs.add_projection_requirement(READ_ONLY, *this, 
                                  SNAP_ENERGY_GROUP_FIELD(group));
  vdelt.add_region_requirement(READ_ONLY, *this, 
                               SNAP_ENERGY_GROUP_FIELD(group));
  dinv.add_projection_requirement(WRITE_DISCARD, *this, 
                                  SNAP_ENERGY_GROUP_FIELD(group));
}

//------------------------------------------------------------------------------
/*static*/ void CalculateGeometryParam::preregister_cpu_variants(void)
//------------------------------------------------------------------------------
{
  register_cpu_variant<cpu_implementation>(true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ void CalculateGeometryParam::preregister_gpu_variants(void)
//------------------------------------------------------------------------------
{
  register_gpu_variant<gpu_implementation>(true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ void CalculateGeometryParam::cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  const int group = *((int*)task->args);
  RegionAccessor<AccessorType::Generic,double> fa_xs = 
    regions[0].get_field_accessor(
        SNAP_ENERGY_GROUP_FIELD(group)).typeify<double>();
  RegionAccessor<AccessorType::Generic> fa_dinv = 
    regions[2].get_field_accessor(SNAP_ENERGY_GROUP_FIELD(group));
  const double vdelt = regions[1].get_field_accessor(
     SNAP_ENERGY_GROUP_FIELD(group)).typeify<double>().read(
      DomainPoint::from_point<1>(Point<1>::ZEROES()));

  Domain dom = runtime->get_index_space_domain(ctx, 
          task->regions[2].region.get_index_space());
  Rect<3> subgrid_bounds = dom.get_rect<3>();

  const size_t buffer_size = Snap::num_angles * sizeof(double);
  double *temp = (double*)malloc(buffer_size);

  for (GenericPointInRectIterator<3> itr(subgrid_bounds); itr; itr++)
  {
    const DomainPoint dp = DomainPoint::from_point<3>(itr.p);
    const double xs = fa_xs.read(dp);
    for (int ang = 0; ang < Snap::num_angles; ang++)
      temp[ang] = 1.0 / (xs + vdelt + Snap::hi * Snap::mu[ang] + 
                         Snap::hj * Snap::eta[ang] + Snap::hk * Snap::xi[ang]);
    fa_dinv.write_untyped(dp, temp, buffer_size);
  }

  free(temp);
#endif
}

//------------------------------------------------------------------------------
/*static*/ void CalculateGeometryParam::gpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  assert(false);
#endif
}

