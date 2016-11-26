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
                         const SnapArray &flux, const SnapArray &fluxm,
                         const SnapArray &qtot, const SnapArray &vdelt, 
                         const SnapArray &dinv, const SnapArray &t_xs,
                         const SnapArray &time_flux_in, 
                         const SnapArray &time_flux_out,
                         int group, int corner, const int ghost_offsets[3])
  : SnapTask<MiniKBATask, Snap::MINI_KBA_TASK_ID>(
      snap, Rect<3>(Point<3>::ZEROES(), Point<3>::ZEROES()), pred), 
    mini_kba_args(MiniKBAArgs(corner, group))
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
  fluxm.add_projection_requirement(Snap::QUAD_REDUCTION_ID, *this,
                                   group_field, Snap::SWEEP_PROJECTION);
  vdelt.add_region_requirement(READ_ONLY, *this, group_field);
  // Add the dinv array for this field
  dinv.add_projection_requirement(READ_ONLY, *this,
                                  group_field, Snap::SWEEP_PROJECTION);
  time_flux_in.add_projection_requirement(READ_ONLY, *this,
                                          group_field, Snap::SWEEP_PROJECTION);
  time_flux_out.add_projection_requirement(WRITE_DISCARD, *this,
                                           group_field, Snap::SWEEP_PROJECTION);
  t_xs.add_projection_requirement(READ_ONLY, *this,
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
    
  // This implementation of the sweep assumes three dimensions
  assert(Snap::num_dims == 3);

  RegionAccessor<AccessorType::Generic,MomentQuad> fa_qtot = 
    regions[0].get_field_accessor(
        SNAP_ENERGY_GROUP_FIELD(args->group)).typeify<MomentQuad>();
  RegionAccessor<AccessorType::Generic,double> fa_flux = 
    regions[1].get_field_accessor(
        SNAP_ENERGY_GROUP_FIELD(args->group)).typeify<double>();
  RegionAccessor<AccessorType::Generic,MomentQuad> fa_fluxm = 
    regions[2].get_field_accessor(
        SNAP_ENERGY_GROUP_FIELD(args->group)).typeify<MomentQuad>();

  // No types here since the size of these fields are dependent
  // on the number of angles
  const double vdelt = regions[3].get_field_accessor(
      SNAP_ENERGY_GROUP_FIELD(args->group)).typeify<double>().read(
      DomainPoint::from_point<1>(Point<1>::ZEROES()));
  RegionAccessor<AccessorType::Generic> fa_dinv = 
    regions[4].get_field_accessor(SNAP_ENERGY_GROUP_FIELD(args->group));
  RegionAccessor<AccessorType::Generic> fa_time_flux_in = 
    regions[5].get_field_accessor(SNAP_ENERGY_GROUP_FIELD(args->group));
  RegionAccessor<AccessorType::Generic> fa_time_flux_out = 
    regions[6].get_field_accessor(SNAP_ENERGY_GROUP_FIELD(args->group));
  RegionAccessor<AccessorType::Generic,double> fa_t_xs = 
    regions[7].get_field_accessor(SNAP_ENERGY_GROUP_FIELD(args->group)).typeify<double>();

  // Output ghost regions
  RegionAccessor<AccessorType::Generic> fa_ghostx_out = 
    regions[MINI_KBA_NON_GHOST_REQUIREMENTS].get_field_accessor(
        SNAP_ENERGY_GROUP_FIELD(args->group));
  RegionAccessor<AccessorType::Generic> fa_ghosty_out = 
    regions[MINI_KBA_NON_GHOST_REQUIREMENTS+1].get_field_accessor(
        SNAP_ENERGY_GROUP_FIELD(args->group));
  RegionAccessor<AccessorType::Generic> fa_ghostz_out = 
    regions[MINI_KBA_NON_GHOST_REQUIREMENTS+2].get_field_accessor(
        SNAP_ENERGY_GROUP_FIELD(args->group));
  // Input ghost regions
  RegionAccessor<AccessorType::Generic> fa_ghostx_in = 
    regions[MINI_KBA_NON_GHOST_REQUIREMENTS+3].get_field_accessor(
        SNAP_ENERGY_GROUP_FIELD(args->group));
  RegionAccessor<AccessorType::Generic> fa_ghosty_in = 
    regions[MINI_KBA_NON_GHOST_REQUIREMENTS+4].get_field_accessor(
        SNAP_ENERGY_GROUP_FIELD(args->group));
  RegionAccessor<AccessorType::Generic> fa_ghostz_in = 
    regions[MINI_KBA_NON_GHOST_REQUIREMENTS+5].get_field_accessor(
        SNAP_ENERGY_GROUP_FIELD(args->group));

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
  const size_t angle_buffer_size = Snap::num_angles * sizeof(double);
  double *psi = (double*)malloc(angle_buffer_size);
  double *pc = (double*)malloc(angle_buffer_size);
  double *psii = (double*)malloc(angle_buffer_size);
  double *psij = (double*)malloc(angle_buffer_size);
  double *psik = (double*)malloc(angle_buffer_size);
  double *time_flux_in = (double*)malloc(angle_buffer_size);
  double *time_flux_out = (double*)malloc(angle_buffer_size);
  double *temp_array = (double*)malloc(angle_buffer_size);
  double *hv_x = (double*)malloc(angle_buffer_size);
  double *hv_y = (double*)malloc(angle_buffer_size);
  double *hv_z = (double*)malloc(angle_buffer_size);
  double *hv_t = (double*)malloc(angle_buffer_size);
  double *fx_hv_x = (double*)malloc(angle_buffer_size);
  double *fx_hv_y = (double*)malloc(angle_buffer_size);
  double *fx_hv_z = (double*)malloc(angle_buffer_size);
  double *fx_hv_t = (double*)malloc(angle_buffer_size);

  typedef std::map<Point<3>,double*,Point<3>::STLComparator> PreviousMap;
  PreviousMap previous_x, previous_y, previous_z;

  const double tolr = 1.0e-12;

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
      if (stride_x_positive) {
        // reading from x-1 
        Point<3> ghost_point = local_point;
        ghost_point.x[0] -= 1;
        if (it->x[0] == 0) {
          // Ghost cell array
          fa_ghostx_in.read_untyped(DomainPoint::from_point<3>(ghost_point),   
                                    psii, angle_buffer_size); 
        } else {
          // Same array
          PreviousMap::iterator finder = previous_x.find(ghost_point);
          assert(finder != previous_x.end());
          free(psii);
          psii = finder->second;
          previous_x.erase(finder);
        }
      } else {
        // reading from x+1
        Point<3> ghost_point = local_point;
        ghost_point.x[0] += 1;
        if (it->x[0] == (Snap::nx-1)) {
          // Ghost cell array
          fa_ghostx_in.read_untyped(DomainPoint::from_point<3>(ghost_point), 
                                    psii, angle_buffer_size);
        } else {
          // Same array
          PreviousMap::iterator finder = previous_x.find(ghost_point);
          assert(finder != previous_x.end());
          free(psii);
          previous_x.erase(finder);
        }
      }
      for (unsigned ang = 0; ang < Snap::num_angles; ang++)
        pc[ang] += psii[ang] * Snap::mu[ang] * Snap::hi;
      // Y ghost cells
      if (stride_y_positive) {
        // reading from y-1
        Point<3> ghost_point = local_point;
        ghost_point.x[1] -= 1;
        if (it->x[1] == 0) {
          // Ghost cell array
          fa_ghosty_in.read_untyped(DomainPoint::from_point<3>(ghost_point), 
                                    psij, angle_buffer_size);
        } else {
          // Same array
          PreviousMap::iterator finder = previous_y.find(ghost_point);
          assert(finder != previous_y.end());
          free(psij);
          psij = finder->second;
          previous_y.erase(finder);
        }
      } else {
        // reading from y+1
        Point<3> ghost_point = local_point;
        ghost_point.x[1] += 1;
        if (it->x[1] == (Snap::ny-1)) {
          // Ghost cell array
          fa_ghosty_in.read_untyped(DomainPoint::from_point<3>(ghost_point), 
                                    psij, angle_buffer_size);
        } else {
          // Same array
          PreviousMap::iterator finder = previous_y.find(ghost_point);
          assert(finder != previous_y.end());
          free(psij);
          psij = finder->second;
          previous_y.erase(finder);
        }
      }
      for (unsigned ang = 0; ang < Snap::num_angles; ang++)
        pc[ang] += psij[ang] * Snap::eta[ang] * Snap::hj;
      // Z ghost cells
      if (stride_z_positive) {
        // reading from z-1
        Point<3> ghost_point = local_point;
        ghost_point.x[2] -= 1;
        if (it->x[2] == 0) {
          // Ghost cell array
          fa_ghostz_in.read_untyped(DomainPoint::from_point<3>(ghost_point), 
                                    psik, angle_buffer_size);
        } else {
          // Same array
          PreviousMap::iterator finder = previous_z.find(ghost_point);
          assert(finder != previous_z.end());
          free(psik);
          psik = finder->second;
          previous_z.erase(finder);
        }
      } else {
        // reading from z+1
        Point<3> ghost_point = local_point;
        ghost_point.x[2] += 1;
        if (it->x[2] == (Snap::nz-1)) {
          // Ghost cell array
          fa_ghostz_in.read_untyped(DomainPoint::from_point<3>(ghost_point), 
                                    psik, angle_buffer_size);
        } else {
          // Same array
          PreviousMap::iterator finder = previous_z.find(ghost_point);
          assert(finder != previous_z.end());
          free(psik);
          psik = finder->second;
          previous_z.erase(finder);
        }
      }
      for (unsigned ang = 0; ang < Snap::num_angles; ang++)
        pc[ang] += psik[ang] * Snap::xi[ang] * Snap::hk;
      // See if we're doing anything time dependent
      if (vdelt != 0.0) 
      {
        fa_time_flux_in.read_untyped(DomainPoint::from_point<3>(local_point),
                                     time_flux_in, angle_buffer_size);
        for (unsigned ang = 0; ang < Snap::num_angles; ang++)
          pc[ang] += vdelt * time_flux_in[ang];
      }
      // Multiple by the precomputed denominator inverse
      fa_dinv.read_untyped(DomainPoint::from_point<3>(local_point),
                           temp_array, angle_buffer_size);
      for (unsigned ang = 0; ang < Snap::num_angles; ang++)
        pc[ang] *= temp_array[ang];

      if (Snap::flux_fixup) {
        // DO THE FIXUP
        unsigned old_negative_fluxes = 0;
        for (unsigned ang = 0; ang < Snap::num_angles; ang++)
          hv_x[ang] = 1.0;
        for (unsigned ang = 0; ang < Snap::num_angles; ang++)
          hv_y[ang] = 1.0;
        for (unsigned ang = 0; ang < Snap::num_angles; ang++)
          hv_z[ang] = 1.0;
        for (unsigned ang = 0; ang < Snap::num_angles; ang++)
          hv_t[ang] = 1.0;
        const double t_xs = fa_t_xs.read(DomainPoint::from_point<3>(local_point));
        while (true) {
          unsigned negative_fluxes = 0;
          // Figure out how many negative fluxes we have
          for (unsigned ang = 0; ang < Snap::num_angles; ang++) {
            fx_hv_x[ang] = 2.0 * pc[ang] - psii[ang];
            if (fx_hv_x[ang] < 0.0) {
              hv_x[ang] = 0.0;
              negative_fluxes++;
            }
          }
          for (unsigned ang = 0; ang < Snap::num_angles; ang++) {
            fx_hv_y[ang] = 2.0 * pc[ang] - psij[ang];
            if (fx_hv_y[ang] < 0.0) {
              hv_y[ang] = 0.0;
              negative_fluxes++;
            }
          }
          for (unsigned ang = 0; ang < Snap::num_angles; ang++) {
            fx_hv_z[ang] = 2.0 * pc[ang] - psik[ang];
            if (fx_hv_z[ang] < 0.0) {
              hv_z[ang] = 0.0;
              negative_fluxes++;
            }
          }
          if (vdelt != 0.0) {
            for (unsigned ang = 0; ang < Snap::num_angles; ang++) {
              fx_hv_t[ang] = 2.0 * pc[ang] - time_flux_in[ang];
              if (fx_hv_t[ang] < 0.0) {
                hv_t[ang] = 0.0;
                negative_fluxes++;
              }
            }
          }
          if (negative_fluxes == old_negative_fluxes)
            break;
          old_negative_fluxes = negative_fluxes; 
          if (vdelt != 0.0) {
            for (unsigned ang = 0; ang < Snap::num_angles; ang++) {
              pc[ang] = psi[ang] + 0.5 * (
                  psii[ang] * Snap::mu[ang] * Snap::hi * (1.0 + hv_x[ang]) + 
                  psij[ang] * Snap::eta[ang] * Snap::hj * (1.0 + hv_y[ang]) + 
                  psik[ang] * Snap::xi[ang] * Snap::hk * (1.0 + hv_z[ang]) + 
                  time_flux_in[ang] * vdelt * (1.0 + hv_t[ang]) );
              double den = (pc[ang] <= 0.0) ? 0 : t_xs + 
                Snap::mu[ang] * Snap::hi * hv_x[ang] + 
                Snap::eta[ang] * Snap::hj * hv_y[ang] +
                Snap::xi[ang] * Snap::hk * hv_z[ang] + vdelt * hv_t[ang];
              if (den < tolr)
                pc[ang] = 0.0;
              else
                pc[ang] /= den;
            }
          } else {
            for (unsigned ang = 0; ang < Snap::num_angles; ang++) {
              pc[ang] = psi[ang] + 0.5 * (
                  psii[ang] * Snap::mu[ang] * Snap::hi * (1.0 + hv_x[ang]) + 
                  psij[ang] * Snap::eta[ang] * Snap::hj * (1.0 + hv_y[ang]) +
                  psik[ang] * Snap::xi[ang] * Snap::hk * (1.0 + hv_z[ang]) );
              double den = (pc[ang] <= 0.0) ? 0 : t_xs + 
                Snap::mu[ang] * Snap::hi * hv_x[ang] + 
                Snap::eta[ang] * Snap::hj * hv_y[ang] +
                Snap::xi[ang] * Snap::hk * hv_z[ang];
              if (den < tolr)
                pc[ang] = 0.0;
              else
                pc[ang] /= den;
            }
          }
        }
        // Fixup done so compute the updated values
        for (unsigned ang = 0; ang < Snap::num_angles; ang++)
          psii[ang] = fx_hv_x[ang] * hv_x[ang];
        for (unsigned ang = 0; ang < Snap::num_angles; ang++)
          psij[ang] = fx_hv_y[ang] * hv_y[ang];
        for (unsigned ang = 0; ang < Snap::num_angles; ang++)
          psik[ang] = fx_hv_z[ang] * hv_z[ang];
        if (vdelt != 0.0)
        {
          for (unsigned ang = 0; ang < Snap::num_angles; ang++)
            time_flux_out[ang] = fx_hv_t[ang] * hv_t[ang];
          fa_time_flux_out.write_untyped(DomainPoint::from_point<3>(local_point),
                                         time_flux_out, angle_buffer_size);
        }
      } else {
        // NO FIXUP
        for (unsigned ang = 0; ang < Snap::num_angles; ang++)
          psii[ang] = 2.0 * pc[ang] - psii[ang]; 
        for (unsigned ang = 0; ang < Snap::num_angles; ang++)
          psij[ang] = 2.0 * pc[ang] - psij[ang];
        for (unsigned ang = 0; ang < Snap::num_angles; ang++)
          psik[ang] = 2.0 * pc[ang] - psik[ang];
        if (vdelt != 0.0) 
        {
          // Write out the outgoing temporal flux
          for (unsigned ang = 0; ang < Snap::num_angles; ang++)
            time_flux_out[ang] = 2.0 * pc[ang] - time_flux_in[ang];
          fa_time_flux_out.write_untyped(DomainPoint::from_point<3>(local_point),
                                         time_flux_out, angle_buffer_size);
        }
      }
      // Write out the ghost regions 
      // X ghost
      if (stride_x_positive) {
        // Writing to x+1
        if (it->x[0] == (Snap::nx-1)) {
          // Write to the ghost cell region
          Point<3> ghost_point = local_point;
          ghost_point.x[0] += 1;   
          fa_ghostx_out.write_untyped(DomainPoint::from_point<3>(ghost_point),
                                      psii, angle_buffer_size);
        } else {
          // Write to the local set
          previous_x[local_point] = psii;
          psii = (double*)malloc(angle_buffer_size);
        }
      } else {
        // Writing to x-1
        if (it->x[0] == 0) {
          // Write to the ghost cell region
          Point<3> ghost_point = local_point;
          ghost_point.x[0] -= 1;
          fa_ghostx_out.write_untyped(DomainPoint::from_point<3>(ghost_point),
                                      psii, angle_buffer_size);
        } else {
          // Write to the local set
          previous_x[local_point] = psii;
          psii = (double*)malloc(angle_buffer_size);
        }
      }
      // Y ghost
      if (stride_y_positive) {
        // Writing to y+1
        if (it->x[1] == (Snap::ny-1)) {
          // Write to the ghost cell region
          Point<3> ghost_point = local_point;
          ghost_point.x[1] += 1;
          fa_ghosty_out.write_untyped(DomainPoint::from_point<3>(ghost_point),
                                      psij, angle_buffer_size);
        } else {
          // Write to the local set
          previous_y[local_point] = psij;
          psij = (double*)malloc(angle_buffer_size);
        }
      } else {
        // Writing to y-1
        if (it->x[1] == 0) {
          // Write to the ghost cell region
          Point<3> ghost_point = local_point;
          ghost_point.x[1] -= 1;
          fa_ghosty_out.write_untyped(DomainPoint::from_point<3>(ghost_point),
                                      psij, angle_buffer_size);
        } else {
          // Write to the local set
          previous_y[local_point] = psij;
          psij = (double*)malloc(angle_buffer_size);
        }
      }
      // Z ghost
      if (stride_z_positive) {
        // Writing to z+1
        if (it->x[2] == (Snap::nz-1)) {
          // Write to the ghost cell region
          Point<3> ghost_point = local_point;
          ghost_point.x[2] += 1;
          fa_ghostz_out.write_untyped(DomainPoint::from_point<3>(ghost_point),
                                      psik, angle_buffer_size);
        } else {
          // Write to the local set
          previous_z[local_point] = psik;
          psik = (double*)malloc(angle_buffer_size);
        }
      } else {
        if (it->x[2] == 0) {
          // Write to the ghost cell region
          Point<3> ghost_point = local_point;
          ghost_point.x[2] -= 1;
          fa_ghostz_out.write_untyped(DomainPoint::from_point<3>(ghost_point),
                                      psik, angle_buffer_size);
        } else {
          // Write to the local set
          previous_z[local_point] = psik;
          psik = (double*)malloc(angle_buffer_size);
        }
      }
      // Finally we apply reductions to the flux moments
      double total = 0.0;
      for (unsigned ang = 0; ang < Snap::num_angles; ang++) {
        psi[ang] = Snap::w[ang] * pc[ang]; 
        total += psi[ang];
      }
      fa_flux.reduce<SumReduction>(DomainPoint::from_point<3>(local_point), total);
      if (Snap::num_moments > 1) {
        MomentQuad quad;
        for (int l = 1; l < Snap::num_moments; l++) {
          unsigned offset = l * Snap::num_angles + 
            args->corner * Snap::num_angles * Snap::num_moments;
          total = 0.0;
          for (unsigned ang = 0; ang < Snap::num_angles; ang++) {
            total += Snap::ec[offset+ang] * psi[ang]; 
          }
          quad[l] = total;
        }
        fa_fluxm.reduce<QuadReduction>(DomainPoint::from_point<3>(local_point), quad);
      }
    }
  }

  free(psi);
  free(pc);
  free(psii);
  free(psij);
  free(psik);
  free(time_flux_in);
  free(time_flux_out);
  free(temp_array);
  free(hv_x);
  free(hv_y);
  free(hv_z);
  free(hv_t);
  free(fx_hv_x);
  free(fx_hv_y);
  free(fx_hv_z);
  free(fx_hv_t);
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

