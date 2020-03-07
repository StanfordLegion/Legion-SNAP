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
#include "init.h"

extern Legion::Logger log_snap;

//------------------------------------------------------------------------------
InitMaterial::InitMaterial(const Snap &snap, const SnapArray<3> &mat)
  : SnapTask<InitMaterial,Snap::INIT_MATERIAL_TASK_ID>(
      snap, snap.get_launch_bounds(), Predicate::TRUE_PRED)
//------------------------------------------------------------------------------
{
  mat.add_projection_requirement(READ_WRITE, *this);
}

//------------------------------------------------------------------------------
/*static*/ void InitMaterial::preregister_cpu_variants(void)
//------------------------------------------------------------------------------
{
  ExecutionConstraintSet execution_constraints;
  // Need x86 CPU
  execution_constraints.add_constraint(ISAConstraint(X86_ISA));
  TaskLayoutConstraintSet layout_constraints;
  layout_constraints.add_layout_constraint(0/*idx*/,
                                           Snap::get_soa_layout());
  register_cpu_variant<cpu_implementation>(execution_constraints,
                                           layout_constraints,
                                           true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ void InitMaterial::cpu_implementation(const Task *task,
    const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  log_snap.info("Running Init Material");

  int i1 = 1, i2 = 1, j1 = 1, j2 = 1, k1 = 1, k2 = 1;
  switch (Snap::material_layout)
  {
    case Snap::CENTER_LAYOUT:
      {
        const int nx_gl = Snap::nx;
        i1 = nx_gl / 4 + 1;
        i2 = 3 * nx_gl / 4;
        if (Snap::num_dims > 1) {
          const int ny_gl = Snap::ny;
          j1 = ny_gl/ 4 + 1;
          j2 = 3 * ny_gl / 4;
          if (Snap::num_dims > 2) {
            const int nz_gl = Snap::nz;
            k1 = nz_gl / 4 + 1;
            k2 = 3 * nz_gl / 4;
          }
        }
        break;
      }
    case Snap::CORNER_LAYOUT:
      {
        const int nx_gl = Snap::nx;
        i2 = nx_gl / 2;
        if (Snap::num_dims > 1) {
          const int ny_gl = Snap::ny;
          j2 = ny_gl / 2;
          if (Snap::num_dims > 2) {
            const int nz_gl = Snap::nz;
            k2 = nz_gl / 2;
          }
        }
        break;
      }
    default:
      assert(false);
  }
  Domain<3> dom = runtime->get_index_space_domain(ctx, 
          IndexSpace<3>(task->regions[0].region.get_index_space()));
  AccessorRW<int,3> fa_mat(regions[0], Snap::FID_SINGLE);
  Rect<3> mat_bounds(Point<3>(i1-1, j1-1, k1-1),
                     Point<3>(i2-1, j2-1, k2-1));;
  Rect<3> local_bounds = dom.bounds.intersection(mat_bounds);
  if (local_bounds.volume() == 0)
    return;
  for (RectIterator<3> itr(local_bounds); itr(); itr++)
    fa_mat[*itr] = 2;
#endif
}

//------------------------------------------------------------------------------
InitSource::InitSource(const Snap &snap, const SnapArray<3> &qi)
  : SnapTask<InitSource, Snap::INIT_SOURCE_TASK_ID>(
      snap, snap.get_launch_bounds(), Predicate::TRUE_PRED)
//------------------------------------------------------------------------------
{
  qi.add_projection_requirement(READ_WRITE, *this);
}

//------------------------------------------------------------------------------
/*static*/ void InitSource::preregister_cpu_variants(void)
//------------------------------------------------------------------------------
{
  ExecutionConstraintSet execution_constraints;
  // Need x86 CPU
  execution_constraints.add_constraint(ISAConstraint(X86_ISA));
  TaskLayoutConstraintSet layout_constraints;
  layout_constraints.add_layout_constraint(0/*index*/,
                                           Snap::get_soa_layout());
  register_cpu_variant<cpu_implementation>(execution_constraints,
                                           layout_constraints,
                                           true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ void InitSource::cpu_implementation(const Task *task,
    const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  log_snap.info("Running Init Source");

  const int nx_gl = Snap::nx;
  const int ny_gl = Snap::ny;
  const int nz_gl = Snap::nz;

  int i1 = 1, i2 = nx_gl, j1 = 1, j2 = ny_gl, k1 = 1, k2 = nz_gl;

  switch (Snap::source_layout)
  {
    case Snap::EVERYWHERE_SOURCE:
      break;
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
    default: // nothing else should be called
      assert(false);
  }
  Domain<3> dom = runtime->get_index_space_domain(ctx, 
          IndexSpace<3>(task->regions[0].region.get_index_space()));
  Rect<3> source_bounds(Point<3>(i1-1, j1-1, k1-1),
                        Point<3>(i2-1, j2-1, k2-1));;
  Rect<3> local_bounds = dom.bounds.intersection(source_bounds);
  if (local_bounds.volume() == 0)
    return;
  for (std::set<FieldID>::const_iterator it = 
        task->regions[0].privilege_fields.begin(); it !=
        task->regions[0].privilege_fields.end(); it++)
  {
    AccessorRW<double,3> fa_qi(regions[0], *it);
    for (RectIterator<3> itr(local_bounds); itr(); itr++)
      fa_qi[*itr] = 1.0;
  }
#endif
}

//------------------------------------------------------------------------------
InitScattering::InitScattering(const SnapArray<1> &sigt,
                               const SnapArray<1> &siga,
                               const SnapArray<1> &sigs,
                               const SnapArray<2> &slgg)
  : TaskLauncher(TASK_ID, TaskArgument())
//------------------------------------------------------------------------------
{
  sigt.add_region_requirement(READ_WRITE, *this);
  siga.add_region_requirement(READ_WRITE, *this);
  sigs.add_region_requirement(READ_WRITE, *this);
  slgg.add_region_requirement(READ_WRITE, *this);
  // All SNAP tasks need 3-D points
  point = Point<3>(0,0,0);
}

//------------------------------------------------------------------------------
void InitScattering::dispatch(Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
  runtime->execute_task(ctx, *this);
}

//------------------------------------------------------------------------------
/*static*/ void InitScattering::preregister_cpu_variants(void)
//------------------------------------------------------------------------------
{
  char variant_name[128];
  strcpy(variant_name, "CPU ");
  strncat(variant_name, Snap::task_names[TASK_ID], 123);
  TaskVariantRegistrar registrar(TASK_ID, true/*global*/, variant_name);
  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  registrar.leaf_variant = true;
  registrar.inner_variant = false;
  for (int idx = 0; idx < 4; idx++)
    registrar.layout_constraints.add_layout_constraint(idx, 
                                    Snap::get_soa_layout());
  Runtime::preregister_task_variant<cpu_implementation>(
                                registrar, Snap::task_names[TASK_ID]);
}

//------------------------------------------------------------------------------
/*static*/ void InitScattering::cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
  PhysicalRegion sigt_region = regions[0];
  PhysicalRegion siga_region = regions[1];
  PhysicalRegion sigs_region = regions[2];
  PhysicalRegion slgg_region = regions[3];

  std::vector<AccessorRW<double,1> > fa_sigt(Snap::num_groups);
  std::vector<AccessorRW<double,1> > fa_siga(Snap::num_groups);
  std::vector<AccessorRW<double,1> > fa_sigs(Snap::num_groups);
  for (int g = 0; g < Snap::num_groups; g++)
  {
    fa_sigt[g] = AccessorRW<double,1>(sigt_region, SNAP_ENERGY_GROUP_FIELD(g));
    fa_siga[g] = AccessorRW<double,1>(siga_region, SNAP_ENERGY_GROUP_FIELD(g));
    fa_sigs[g] = AccessorRW<double,1>(sigs_region, SNAP_ENERGY_GROUP_FIELD(g));
  }

  fa_sigt[0][1] = 1.0; 
  fa_siga[0][1] = 0.5;
  fa_sigs[0][1] = 0.5;
  for (int g = 1; g < Snap::num_groups; g++)
  {
    fa_sigt[g][1] = 0.01 * fa_sigt[g-1][1];
    fa_siga[g][1] = 0.005 * fa_siga[g-1][1];
    fa_sigs[g][1] = 0.005 * fa_sigs[g-1][1];
  }

  if (Snap::material_layout != Snap::HOMOGENEOUS_LAYOUT) {
    fa_sigt[0][2] = 2.0;
    fa_siga[0][2] = 0.8;
    fa_sigs[0][2] = 1.2;
    for (int g = 1; g < Snap::num_groups; g++)
    {
      fa_sigt[g][2] = 0.01 * fa_sigt[g-1][2];
      fa_siga[g][2] = 0.005 * fa_siga[g-1][2];
      fa_sigs[g][2] = 0.005 * fa_sigs[g-1][2];
    }
  }

  std::vector<AccessorRW<MomentQuad,2> > fa_slgg(Snap::num_groups); 
  for (int g = 0; g < Snap::num_groups; g++)
    fa_slgg[g] = AccessorRW<MomentQuad,2>(slgg_region, 
                          SNAP_ENERGY_GROUP_FIELD(g));

  if (Snap::num_groups == 1) {
    MomentQuad local;
    local[0] = fa_sigs[0][1];
    fa_slgg[0][1][0] = local;
    if (Snap::material_layout != Snap::HOMOGENEOUS_LAYOUT) {
      local[0] = fa_sigs[0][2];
      fa_slgg[0][1][1] = local;
    }
  } else {
    MomentQuad local;
    for (int g = 0; g < Snap::num_groups; g++) {
      local[0] = 0.2 * fa_sigs[g][1];
      fa_slgg[g][1][g] = local;
      if (g > 0) {
        const double t = 1.0 / double(g);
        for (int g2 = 0; g2 < g; g2++) {
          local[0] = 0.1 * fa_sigs[g][1] * t;
          fa_slgg[g2][1][g] = local;
        }
      } else {
        local[0] = 0.3 * fa_sigs[g][1];
        fa_slgg[g][1][g] = local;
      }

      if (g < (Snap::num_groups-1)) {
        const double t = 1.0 / double(Snap::num_groups-(g+1));
        for (int g2 = g+1; g2 < Snap::num_groups; g2++) {
          local[0] = 0.7 * fa_sigs[g][1] * t;
          fa_slgg[g2][1][g] = local;
        }
      } else {
        local[0] = 0.9 * fa_sigs[g][1];
        fa_slgg[g][1][g] = local;
      }
    }
    if (Snap::material_layout != Snap::HOMOGENEOUS_LAYOUT) {
      for (int g = 0; g < Snap::num_groups; g++) {
        local[0] = 0.5 * fa_sigs[g][2];
        fa_slgg[g][2][g] = local;
        if (g > 0) {
          const double t = 1.0 / double(g);
          for (int g2 = 0; g2 < g; g2++) {
            local[0] = 0.1 * fa_sigs[g][2] * t;
            fa_slgg[g2][2][g] = local;
          }
        } else {
          local[0] = 0.6 * fa_sigs[g][2];
          fa_slgg[g][2][g] = local;
        }

        if (g < (Snap::num_groups-1)) {
          const double t = 1.0 / double(Snap::num_groups-(g+1));
          for (int g2 = g+1; g2 < Snap::num_groups; g2++) {
            local[0] = 0.4 * fa_sigs[g][2] * t;
            fa_slgg[g2][2][g] = local;
          }
        } else {
          local[0] = 0.9 * fa_sigs[g][2];
          fa_slgg[g][2][g] = local;
        }
      }
    }
  }
  if (Snap::num_moments > 1) 
  {
    for (int m = 1; m < Snap::num_moments; m++) {
      for (int g = 0; g < Snap::num_groups; g++) {
        for (int g2 = 0; g2 < Snap::num_groups; g2++) {
          MomentQuad quad = fa_slgg[g2][1][g];
          quad[m] = ((m == 1) ? 0.1 : 0.5) * quad[m-1];
          fa_slgg[g2][1][g] = quad;
        }
      }
    }
    if (Snap::material_layout != Snap::HOMOGENEOUS_LAYOUT) {
      for (int m = 1; m < Snap::num_moments; m++) {
        for (int g = 0; g < Snap::num_groups; g++) {
          for (int g2 = 0; g2 < Snap::num_groups; g2++) {
            MomentQuad quad = fa_slgg[g2][2][g];
            quad[m] = ((m == 1) ? 0.8 : 0.6) * quad[m-1];
            fa_slgg[g2][2][g] = quad;
          }
        }
      }
    }
  }
}

//------------------------------------------------------------------------------
InitVelocity::InitVelocity(const SnapArray<1> &vel,
                           const SnapArray<1> &vdelt)
  : TaskLauncher(TASK_ID, TaskArgument())
//------------------------------------------------------------------------------
{
  vel.add_region_requirement(READ_WRITE, *this);
  vdelt.add_region_requirement(READ_WRITE, *this);
  // All SNAP tasks need 3-D points
  point = Point<3>(0,0,0);
}

//------------------------------------------------------------------------------
void InitVelocity::dispatch(Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
  runtime->execute_task(ctx, *this);
}

//------------------------------------------------------------------------------
/*static*/ void InitVelocity::preregister_cpu_variants(void)
//------------------------------------------------------------------------------
{
  char variant_name[128];
  strcpy(variant_name, "CPU ");
  strncat(variant_name, Snap::task_names[TASK_ID], 123);
  TaskVariantRegistrar registrar(TASK_ID, true/*global*/, variant_name);
  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  registrar.leaf_variant = true;
  registrar.inner_variant = false;
  for (int idx = 0; idx < 2; idx++)
    registrar.layout_constraints.add_layout_constraint(idx, 
                                    Snap::get_soa_layout());
  Runtime::preregister_task_variant<cpu_implementation>(
                                registrar, Snap::task_names[TASK_ID]);
}

//------------------------------------------------------------------------------
/*static*/ void InitVelocity::cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
  PhysicalRegion vel_region = regions[0];
  PhysicalRegion vdelt_region = regions[1];

  const Point<1> dp(0);
  for (int g = 0; g < Snap::num_groups; g++) 
  {
    AccessorRW<double,1> fa_vel(vel_region, SNAP_ENERGY_GROUP_FIELD(g));
    AccessorRW<double,1> fa_vdelt(vdelt_region, SNAP_ENERGY_GROUP_FIELD(g));
    const double v = double(Snap::num_groups - g);
    fa_vel[dp] = v;
    if (Snap::time_dependent)
      fa_vdelt[dp] = 2.0 / (Snap::dt * v);
    else
      fa_vdelt[dp] = 0.0;
  }
}

//------------------------------------------------------------------------------
InitGPUSweep::InitGPUSweep(const Snap &snap, const Rect<3> &launch)
  : SnapTask<InitGPUSweep, Snap::INIT_GPU_SWEEP_TASK_ID>(
      snap, launch, Predicate::TRUE_PRED)
//------------------------------------------------------------------------------
{
}

//------------------------------------------------------------------------------
/*static*/ void InitGPUSweep::preregister_gpu_variants(void)
//------------------------------------------------------------------------------
{
  ExecutionConstraintSet execution_constraints;
  // Need a CUDA GPU with at least sm30
  execution_constraints.add_constraint(ISAConstraint(CUDA_ISA));
  TaskLayoutConstraintSet layout_constraints;
  register_gpu_variant<gpu_implementation>(execution_constraints,
                                           layout_constraints,
                                           true/*leaf*/);
}

#ifdef USE_GPU_KERNELS
extern void initialize_gpu_context(const double *ec_h, const double *mu_h,
                                   const double *eta_h, const double *xi_h,
                                   const double *w_h, const int num_angles,
                                   const int num_moments, const int num_octants,
                                   const int nx_per_chunk, const int ny_per_chunk,
                                   const int nz_per_chunk);
#endif

//------------------------------------------------------------------------------
/*static*/ void InitGPUSweep::gpu_implementation(const Task *task,
    const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime) 
//------------------------------------------------------------------------------
{
  log_snap.info("Running Init GPU Sweep");
#ifdef USE_GPU_KERNELS
  initialize_gpu_context(Snap::ec, Snap::mu, Snap::eta, Snap::xi, Snap::w,
                         Snap::num_angles, Snap::num_moments, Snap::num_octants,
                         Snap::nx_per_chunk, Snap::ny_per_chunk, Snap::nz_per_chunk);
#else
  assert(false); 
#endif
}

