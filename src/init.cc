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
#include "init.h"

using namespace LegionRuntime::Accessor;

//------------------------------------------------------------------------------
InitMaterial::InitMaterial(const Snap &snap, const SnapArray &mat)
  : SnapTask<InitMaterial>(snap, snap.get_launch_bounds(), Predicate::TRUE_PRED)
//------------------------------------------------------------------------------
{
  mat.add_projection_requirement(READ_WRITE, *this);
}

//------------------------------------------------------------------------------
/*static*/ void InitMaterial::preregister_cpu_variants(void)
//------------------------------------------------------------------------------
{
  register_cpu_variant<cpu_implementation>(true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ void InitMaterial::cpu_implementation(const Task *task,
    const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  int i1 = 1, i2 = 1, j1 = 1, j2 = 1, k1 = 1, k2 = 1;
  switch (Snap::material_layout)
  {
    case Snap::CENTER_LAYOUT:
      {
        const int nx_gl = Snap::nx * Snap::nx_chunks;
        i1 = nx_gl / 4 + 1;
        i2 = 3 * nx_gl / 4;
        if (Snap::num_dims > 1) {
          const int ny_gl = Snap::ny * Snap::ny_chunks;
          j1 = ny_gl/ 4 + 1;
          j2 = 3 * ny_gl / 4;
          if (Snap::num_dims > 2) {
            const int nz_gl = Snap::nz * Snap::nz_chunks;
            k1 = nz_gl / 4 + 1;
            k2 = 3 * nz_gl / 4;
          }
        }
        break;
      }
    case Snap::CORNER_LAYOUT:
      {
        const int nx_gl = Snap::nx * Snap::nx_chunks;
        i2 = nx_gl / 2;
        if (Snap::num_dims > 1) {
          const int ny_gl = Snap::ny * Snap::ny_chunks;
          j2 = ny_gl / 2;
          if (Snap::num_dims > 2) {
            const int nz_gl = Snap::nz * Snap::nz_chunks;
            k2 = nz_gl / 2;
          }
        }
        break;
      }
    default:
      assert(false);
  }
  Domain dom = runtime->get_index_space_domain(ctx, 
          task->regions[0].region.get_index_space());
  Rect<3> subgrid_bounds = dom.get_rect<3>();
  RegionAccessor<AccessorType::Generic,int> fa_mat = 
    regions[0].get_field_accessor(Snap::FID_SINGLE).typeify<int>();
  Rect<3> mat_bounds;
  mat_bounds.lo.x[0] = i1-1;
  mat_bounds.lo.x[1] = j1-1;
  mat_bounds.lo.x[2] = k1-1;
  mat_bounds.hi.x[0] = i2-1;
  mat_bounds.hi.x[1] = j2-1;
  mat_bounds.hi.x[2] = k2-1;
  Rect<3> local_bounds = subgrid_bounds.intersection(mat_bounds);
  if (local_bounds.volume() == 0)
    return;
  for (GenericPointInRectIterator<3> itr(local_bounds); itr; itr++) {
    DomainPoint dp = DomainPoint::from_point<3>(itr.p);
    fa_mat.write(dp, 2);
  }
#endif
}

//------------------------------------------------------------------------------
InitSource::InitSource(const Snap &snap, const SnapArray &qi)
  : SnapTask<InitSource>(snap, snap.get_launch_bounds(), Predicate::TRUE_PRED)
//------------------------------------------------------------------------------
{
  qi.add_projection_requirement(READ_WRITE, *this);
}

//------------------------------------------------------------------------------
/*static*/ void InitSource::preregister_cpu_variants(void)
//------------------------------------------------------------------------------
{
  register_cpu_variant<cpu_implementation>(true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ void InitSource::cpu_implementation(const Task *task,
    const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  const int nx_gl = Snap::nx * Snap::nx_chunks;
  const int ny_gl = Snap::ny * Snap::ny_chunks;
  const int nz_gl = Snap::nz * Snap::nz_chunks;

  int i1 = 1, i2 = nx_gl, j1 = 1, j2 = ny_gl, k1 = 1, k2 = nz_gl;

  switch (Snap::source_layout)
  {
    case Snap::CENTER_SOURCE:
      {
        i1 = nx_gl / 4 + 1;
        i2 = 3 * nx_gl / 4;
        if (Snap::num_dims > 1) {
          j1 = ny_gl / 4 + 1;
          j2 = 3 * ny_gl / 4;
          if (Snap::num_dims > 2) { 
            k1 = nz_gl / 4 + 1;
            k2 = 3 * nz_gl / 4;
          }
        }
        break;
      }
    case Snap::CORNER_SOURCE:
      {
        i2 = nx_gl / 2;
        if (Snap::num_dims > 1) {
          j2 = ny_gl / 2;
          if (Snap::num_dims > 2)
            k2 = nz_gl / 2;
        }
        break;
      }
    case Snap::MMS_SOURCE:
      {
        // TODO: Implement this
        assert(false);
        break;
      }
    default:
      assert(false);
  }
  Domain dom = runtime->get_index_space_domain(ctx, 
          task->regions[0].region.get_index_space());
  Rect<3> subgrid_bounds = dom.get_rect<3>();
  Rect<3> source_bounds;
  source_bounds.lo.x[0] = i1-1;
  source_bounds.lo.x[1] = j1-1;
  source_bounds.lo.x[2] = k1-1;
  source_bounds.hi.x[0] = i2-1;
  source_bounds.hi.x[1] = j2-1;
  source_bounds.hi.x[2] = k2-1;
  Rect<3> local_bounds = subgrid_bounds.intersection(source_bounds);
  if (local_bounds.volume() == 0)
    return;
  for (std::set<FieldID>::const_iterator it = 
        task->regions[0].privilege_fields.begin(); it !=
        task->regions[0].privilege_fields.end(); it++)
  {
    RegionAccessor<AccessorType::Generic,double> fa_qi = 
      regions[0].get_field_accessor(*it).typeify<double>();
    for (GenericPointInRectIterator<3> itr(local_bounds); itr; itr++) {
      DomainPoint dp = DomainPoint::from_point<3>(itr.p);
      fa_qi.write(dp, 1.0);
    }
  }
#endif
}

