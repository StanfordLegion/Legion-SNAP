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
#include "sweep.h"

using namespace LegionRuntime::Accessor;

//------------------------------------------------------------------------------
MiniKBATask::MiniKBATask(const Snap &snap, const Predicate &pred, bool even,
                         const SnapArray &flux, const SnapArray &qtot,
                         int group, int corner, const int ghost_offsets[3])
  : SnapTask<MiniKBATask>(snap, Rect<3>(Point<3>::ZEROES(), Point<3>::ZEROES()),
                          pred), mini_kba_args(MiniKBAArgs(corner, group))
//------------------------------------------------------------------------------
{
  global_arg = TaskArgument(&mini_kba_args, sizeof(mini_kba_args));
  Snap::SnapFieldID group_field = SNAP_ENERGY_GROUP_FIELD(group);
  // If you add projection requirements here, remember to update
  // the value of NON_GHOST_REQUIREMENTS in sweep.h
  qtot.add_projection_requirement(READ_ONLY, *this, 
                                  group_field, Snap::SWEEP_PROJECTION);
  // We need reduction privileges on the flux field since all sweeps
  // will be contributing to it
  flux.add_projection_requirement(Snap::SUM_REDUCTION_ID, *this, 
                                  group_field, Snap::SWEEP_PROJECTION);
  // Then add our writing ghost regions
  for (int i = 0; i < Snap::num_dims; i++)
  {
    Snap::SnapFieldID ghost_write = even ? 
      SNAP_GHOST_FLUX_FIELD_EVEN(group, corner, i) :
      SNAP_GHOST_FLUX_FIELD_ODD(group, corner, i);
    flux.add_projection_requirement(WRITE_DISCARD, *this, 
                                    ghost_write, Snap::SWEEP_PROJECTION);
  }
  assert(region_requirements.size() <= MINI_KBA_NON_GHOST_REQUIREMENTS);
  // Add our reading ghost regions
  for (int i = 0; i < Snap::num_dims; i++)
  {
    // Reverse polarity for these ghost fields
    Snap::SnapFieldID ghost_read = even ?
      SNAP_GHOST_FLUX_FIELD_ODD(group, corner, i) :
      SNAP_GHOST_FLUX_FIELD_EVEN(group, corner, i);
    // We know our projection ID now
    Snap::SnapProjectionID proj_id = SNAP_GHOST_PROJECTION(i, ghost_offsets[i]);
    flux.add_projection_requirement(READ_ONLY, *this, ghost_read, proj_id);
  }
}

//------------------------------------------------------------------------------
void MiniKBATask::dispatch_wavefront(int wavefront, const Domain &launch_d,
                                     Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
  // Save our wavefront
  this->mini_kba_args.wavefront = wavefront;
  // Set our launch domain
  this->launch_domain = launch_d;
  // Then call the normal dispatch routine
  dispatch(ctx, runtime);
}

//------------------------------------------------------------------------------
/*static*/ void MiniKBATask::preregister_cpu_variants(void)
//------------------------------------------------------------------------------
{
  register_cpu_variant<cpu_implementation>(true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ void MiniKBATask::preregister_gpu_variants(void)
//------------------------------------------------------------------------------
{
  register_gpu_variant<gpu_implementation>(true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ void MiniKBATask::cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  assert(task->arglen == sizeof(MiniKBAArgs));
  const MiniKBAArgs *args = reinterpret_cast<const MiniKBAArgs*>(task->args);
    
  RegionAccessor<AccessorType::Generic,MomentQuad> fa_qtot = 
    regions[0].get_field_accessor(
        SNAP_ENERGY_GROUP_FIELD(args->group)).typeify<MomentQuad>();
  RegionAccessor<AccessorType::Generic,double> fa_flux = 
    regions[1].get_field_accessor(
        SNAP_ENERGY_GROUP_FIELD(args->group)).typeify<double>();

  Domain dom = runtime->get_index_space_domain(ctx, 
          task->regions[0].region.get_index_space());
  Rect<3> subgrid_bounds = dom.get_rect<3>();

  // Figure out the origin point based on which corner we are
  const bool stride_x_positive = ((args->corner & 0x1) == 0);
  const bool stride_y_positive = ((args->corner & 0x2) == 0);
  const bool stride_z_positive = ((args->corner & 0x4) == 0);
  const int origin_ints[3] = { 
    (stride_x_positive ? subgrid_bounds.lo[0] : subgrid_bounds.hi[0]),
    (stride_y_positive ? subgrid_bounds.lo[1] : subgrid_bounds.hi[1]),
    (stride_z_positive ? subgrid_bounds.lo[2] : subgrid_bounds.hi[2]) };
  const Point<3> origin(origin_ints);

  const unsigned total_wavefronts = Snap::chunk_wavefronts.size();

  // Local arrays
  double *psi = (double*)malloc(Snap::num_angles * sizeof(double));
  double *pc = (double*)malloc(Snap::num_angles * sizeof(double));

  // Inline the computation of vdelt
  const double vdelt = (Snap::time_dependent ? 
    2.0 / (Snap::dt * double(Snap::num_groups - args->group + 1)) : 0.0);

  // Iterate over wavefronts
  for (unsigned wavefront = 0; wavefront < total_wavefronts; wavefront++)
  {
    // Iterate over points in the wavefront
    const std::vector<Point<3> > &points = Snap::chunk_wavefronts[wavefront];
    for (std::vector<Point<3> >::const_iterator it = points.begin();
          it != points.end(); it++)
    {
      // Figure out the local point that we are working on    
      Point<3> local_point = origin;
      if (stride_x_positive)
        local_point.x[0] += it->x[0];
      else
        local_point.x[0] -= it->x[0];
      if (stride_y_positive)
        local_point.x[1] += it->x[1];
      else
        local_point.x[1] -= it->x[1];
      if (stride_z_positive)
        local_point.x[2] += it->x[2];
      else
        local_point.x[2] -= it->x[2];
      // Compute the angular source
      const DomainPoint dp = DomainPoint::from_point<3>(local_point);
      const MomentQuad quad = fa_qtot.read(dp);
      for (unsigned ang = 0; ang < Snap::num_angles; ang++)
        psi[ang] = quad[0];
      if (Snap::num_moments > 1) {
        const int corner_offset = 
          args->corner * Snap::num_angles * Snap::num_moments;
        for (unsigned l = 1; 1 < Snap::num_moments; l++) {
          const int moment_offset = corner_offset + l * Snap::num_angles;
          for (unsigned ang = 0; ang < Snap::num_angles; ang++) {
            psi[ang] += Snap::ec[moment_offset+ang] * quad[l];
          }
        }
      }
      // Compute the initial solution
      for (unsigned ang = 0; ang < Snap::num_angles; ang++)
        pc[ang] = psi[ang];
      // X ghost cells

      // Y ghost cells

      // Z ghost cells
      if (vdelt != 0.0) 
      {
        
      }
      // Dinv

    }
  }

  free(psi);
  free(pc);
#endif
}

//------------------------------------------------------------------------------
/*static*/ void MiniKBATask::gpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  assert(false);
#endif
}

