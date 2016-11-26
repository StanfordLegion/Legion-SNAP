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
#include "outer.h"
#include "inner.h"
#include "sweep.h"
#include "expxs.h"

#include <cstdio>

#define MIN(x,y) (((x) < (y)) ? (x) : (y))
#define MAX(x,y) (((x) > (y)) ? (x) : (y))

using namespace LegionRuntime::Accessor;

LegionRuntime::Logger::Category log_snap("snap");

const char* Snap::task_names[LAST_TASK_ID] = { SNAP_TASK_NAMES };

//------------------------------------------------------------------------------
void Snap::setup(void)
//------------------------------------------------------------------------------
{
  // This is the index space for all our regions
  // To handle vaccum boundary conditions we allocate an extra chunks of cells
  // on each side of the space just to make everything easier for the
  // computation later. Ironically this makes it look more like the fortran
  // implementation because it makes a lot of indexing look 1-based. This
  // has virtually no overhead cause making overly large logical regions
  // doesn't result in any memory allocation.
  int x_cells_per_chunk = nx / nx_chunks;
  int y_cells_per_chunk = ny / ny_chunks;
  int z_cells_per_chunk = nz / nz_chunks;
  const int upper[3] = { (nx_chunks+2)*x_cells_per_chunk - 1,
                         (ny_chunks+2)*y_cells_per_chunk - 1,
                         (nz_chunks+2)*z_cells_per_chunk - 1 };
  simulation_bounds = Rect<3>(Point<3>::ZEROES(), Point<3>(upper));
  simulation_is = 
    runtime->create_index_space(ctx, Domain::from_rect<3>(simulation_bounds));
  runtime->attach_name(simulation_is, "Simulation Space");
  // Create the disjoint partition of the index space 
  const int bf[3] = { x_cells_per_chunk, y_cells_per_chunk, z_cells_per_chunk };
  Point<3> blocking_factor(bf);
  Blockify<3> spatial_map(blocking_factor);
  spatial_ip = 
    runtime->create_index_partition(ctx, simulation_is,
                                    spatial_map, DISJOINT_PARTITION);
  runtime->attach_name(spatial_ip, "Spatial Partition");
  // Launch bounds though ignore the boundary condition chunks
  // so they start at 1 and go to number of chunks, just like Fortran!
  const int chunks[3] = { nx_chunks, ny_chunks, nz_chunks };
  launch_bounds = Rect<3>(Point<3>::ONES(), Point<3>(chunks)); 
  // Create the ghost partitions for each subregion
  Rect<3> color_space(Point<3>::ZEROES(), Point<3>(chunks) + Point<3>::ONES());
  const char *ghost_names[3] = 
    { "Ghost X Partition", "Ghost Y Partition", "Ghost Z Partition" };
  for (GenericPointInRectIterator<3> itr(color_space); itr; itr++)
  {
    IndexSpace child_is = 
      runtime->get_index_subspace<3>(ctx, spatial_ip, itr.p);
    Rect<3> child_bounds = spatial_map.preimage(itr.p);
    for (int i = 0; i < num_dims; i++)
    {
      DomainColoring dc;

      Rect<3> bounds_lo = child_bounds;
      bounds_lo.hi.x[i] = bounds_lo.lo.x[i];
      dc[LO_GHOST] = Domain::from_rect<3>(bounds_lo);

      Rect<3> bounds_hi = child_bounds;
      bounds_hi.lo.x[i] = bounds_hi.hi.x[i];
      dc[HI_GHOST] = Domain::from_rect<3>(bounds_hi);

      IndexPartition ip = runtime->create_index_partition(ctx, child_is, 
                   Domain::from_rect<1>(Rect<1>(LO_GHOST,HI_GHOST)), dc, 
                   DISJOINT_KIND, GHOST_X_PARTITION+i);
      runtime->attach_name(ip, ghost_names[i]);
    }
  }
  // Make some of our other field spaces
  const int nmat = (material_layout == HOMOGENEOUS_LAYOUT) ? 1 : 2;
  material_is = runtime->create_index_space(ctx,
      Domain::from_rect<1>(Rect<1>(Point<1>(0), Point<1>(nmat-1))));
  const int slgg_upper[2] = { nmat-1, num_groups-1 };
  Rect<2> slgg_bounds(Point<2>::ZEROES(), Point<2>(slgg_upper));
  slgg_is = runtime->create_index_space(ctx, Domain::from_rect<2>(slgg_bounds));
  runtime->attach_name(slgg_is, "Scattering Index Space");
  point_is = runtime->create_index_space(ctx, 
      Domain::from_rect<1>(Rect<1>(Point<1>::ZEROES(), Point<1>::ZEROES())));
  runtime->attach_name(point_is, "Point Index Space");
  // Make a field space for all the energy groups
  group_fs = runtime->create_field_space(ctx); 
  runtime->attach_name(group_fs, "Energy Group Field Space");
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, group_fs);
    assert(num_groups <= SNAP_MAX_ENERGY_GROUPS);
    std::vector<FieldID> group_fields(num_groups);
    for (int idx = 0; idx < num_groups; idx++)
      group_fields[idx] = SNAP_ENERGY_GROUP_FIELD(idx);
    std::vector<size_t> group_sizes(num_groups, sizeof(double));
    allocator.allocate_fields(group_sizes, group_fields);
    char name_buffer[64];
    for (int idx = 0; idx < num_groups; idx++)
    {
      snprintf(name_buffer,63,"Energy Group %d", idx);
      runtime->attach_name(group_fs, group_fields[idx], name_buffer);
    }
  }
  // This field space contains all the energy group fields and ghost fields
  group_and_ghost_fs = runtime->create_field_space(ctx);
  runtime->attach_name(group_and_ghost_fs,"Energy Group and Ghost Field Space");
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, group_and_ghost_fs);
    // Normal energy group fields
    assert(num_groups <= SNAP_MAX_ENERGY_GROUPS);
    std::vector<FieldID> group_fields(num_groups);
    for (int idx = 0; idx < num_groups; idx++)
      group_fields[idx] = SNAP_ENERGY_GROUP_FIELD(idx);
    std::vector<size_t> group_sizes(num_groups, sizeof(double));
    allocator.allocate_fields(group_sizes, group_fields);
    char name_buffer[64];
    for (int idx = 0; idx < num_groups; idx++)
    {
      snprintf(name_buffer,63,"Energy Group %d", idx);
      runtime->attach_name(group_and_ghost_fs, group_fields[idx], name_buffer);
    }
    // ghost corner fields
    // For now we don't do any partitioning of the field space for
    // angles so we make the ghost exchange fields the size of the
    // number of angles.
    // TODO: partition field space of angles 
    std::vector<FieldID> ghost_fields(2*num_groups*num_corners*num_dims);
    std::vector<size_t> ghost_sizes(2*num_groups*num_corners*num_dims, 
                                    num_angles*sizeof(double));
    unsigned next = 0;
    for (int even = 0; even < 2; even++)
    {
      if (even == 0)
        for (int group = 0; group < num_groups; group++)
          for (int corner = 0; corner < num_corners; corner++)
            for (int dim = 0; dim < num_dims; dim++)
              ghost_fields[next++] = 
                SNAP_GHOST_FLUX_FIELD_EVEN(group, corner, dim);
      else
        for (int group = 0; group < num_groups; group++)
          for (int corner = 0; corner < num_corners; corner++)
            for (int dim = 0; dim < num_dims; dim++)
              ghost_fields[next++] = 
                SNAP_GHOST_FLUX_FIELD_ODD(group, corner, dim);
    }
    allocator.allocate_fields(ghost_sizes, ghost_fields);
    const char *ghost_field_names[3] = { "Ghost X", "Ghost Y", "Ghost Z" };
    next = 0;
    for (int even = 0; even < 2; even++)
    {
      if (even == 0)
        for (int group = 0; group < num_groups; group++)
          for (int corner = 0; corner < num_corners; corner++)
            for (int dim = 0; dim < num_dims; dim++)
            {
              snprintf(name_buffer,63,"%s Even Flux for Corner %d of Group %d",
                  ghost_field_names[dim], corner, dim);
              runtime->attach_name(group_and_ghost_fs, 
                                   ghost_fields[next++], name_buffer);
            }
      else
        for (int group = 0; group < num_groups; group++)
          for (int corner = 0; corner < num_corners; corner++)
            for (int dim = 0; dim < num_dims; dim++)
            {
              snprintf(name_buffer,63,"%s Odd Flux for Corner %d of Group %d",
                  ghost_field_names[dim], corner, dim);
              runtime->attach_name(group_and_ghost_fs, 
                                   ghost_fields[next++], name_buffer);
            }
    }
  }
  // Make a fields space for the moments for each energy group
  moment_fs = runtime->create_field_space(ctx);
  runtime->attach_name(moment_fs, "Moment Field Space");
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, moment_fs);
    assert(num_groups <= SNAP_MAX_ENERGY_GROUPS);
    std::vector<FieldID> moment_fields(num_groups);
    for (int idx = 0; idx < num_groups; idx++)
      moment_fields[idx] = SNAP_ENERGY_GROUP_FIELD(idx);
    // Notice that the field size is as big as necessary to store all moments
    std::vector<size_t> moment_sizes(num_groups, sizeof(MomentQuad));
    allocator.allocate_fields(moment_sizes, moment_fields);
    char name_buffer[64];
    for (int idx = 0; idx < num_groups; idx++)
    {
      snprintf(name_buffer,63,"Moment Energy Group %d", idx);
      runtime->attach_name(moment_fs, moment_fields[idx], name_buffer);
    }
  }
  flux_moment_fs = runtime->create_field_space(ctx);
  runtime->attach_name(flux_moment_fs, "Flux Moment Field Space");
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, flux_moment_fs);
    assert(num_groups <= SNAP_MAX_ENERGY_GROUPS);
    std::vector<FieldID> moment_fields(num_groups);
    for (int idx = 0; idx < num_groups; idx++)
      moment_fields[idx] = SNAP_ENERGY_GROUP_FIELD(idx);
    // Storing number of moments - 1
    std::vector<size_t> moment_sizes(num_groups,sizeof(MomentTriple)); 
    allocator.allocate_fields(moment_sizes, moment_fields);
    char name_buffer[64];
    for (int idx = 0; idx < num_groups; idx++)
    {
      snprintf(name_buffer,63,"Moment Flux Energy Group %d", idx);
      runtime->attach_name(flux_moment_fs, moment_fields[idx], name_buffer);
    }
  }
  mat_fs = runtime->create_field_space(ctx);
  runtime->attach_name(mat_fs, "Material Field Space");
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, mat_fs);
    allocator.allocate_field(sizeof(int), FID_SINGLE);
    runtime->attach_name(mat_fs, FID_SINGLE, "Material Field");
  }
  angle_fs = runtime->create_field_space(ctx);
  runtime->attach_name(angle_fs, "Denominator Inverse Field Space");
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, angle_fs);
    std::vector<FieldID> dinv_fields(num_groups);
    for (int idx = 0; idx < num_groups; idx++)
      dinv_fields[idx] = SNAP_ENERGY_GROUP_FIELD(idx);
    std::vector<size_t> dinv_sizes(num_groups, 
                                   Snap::num_angles*sizeof(double));
    allocator.allocate_fields(dinv_sizes, dinv_fields);
    char name_buffer[64];
    for (int idx = 0; idx < num_groups; idx++)
    {
      snprintf(name_buffer,63,"Dinv Group %d", idx);
      runtime->attach_name(angle_fs, dinv_fields[idx], name_buffer);
    }
  }
  // Compute the wavefronts for our sweeps
  for (int corner = 0; corner < num_corners; corner++)
  {
    assert(!wavefront_map[corner].empty());
    for (std::vector<std::vector<DomainPoint> >::const_iterator it = 
         wavefront_map[corner].begin(); it != wavefront_map[corner].end(); it++)
    {
      const size_t wavefront_points = it->size();
      assert(wavefront_points > 0);
      wavefront_domains[corner].push_back(Domain::from_rect<1>(
            Rect<1>(Point<1>(0), Point<1>(wavefront_points-1))));
    }
  }
}

//------------------------------------------------------------------------------
void Snap::transport_solve(void)
//------------------------------------------------------------------------------
{
  // Use a tunable variable to decide how far ahead the outer loop gets
  Future outer_runahead_future = 
    runtime->select_tunable_value(ctx, OUTER_RUNAHEAD_TUNABLE);
  // Same thing with the inner loop
  Future inner_runahead_future = 
    runtime->select_tunable_value(ctx, INNER_RUNAHEAD_TUNABLE);

  // Create our important arrays
  SnapArray flux0(simulation_is, spatial_ip, group_and_ghost_fs, 
                  ctx, runtime, "flux0");
  SnapArray flux0po(simulation_is,spatial_ip, group_fs, 
                    ctx, runtime, "flux0po");
  SnapArray flux0pi(simulation_is, spatial_ip, group_fs, 
                    ctx, runtime, "flux0pi");
  SnapArray fluxm(simulation_is, spatial_ip, flux_moment_fs, 
                  ctx, runtime, "fluxm");

  SnapArray qi(simulation_is, spatial_ip, group_fs, ctx, runtime, "qi");
  SnapArray q2grp0(simulation_is, spatial_ip, group_fs, ctx, runtime, "q2grp0");
  SnapArray q2grpm(simulation_is, spatial_ip, flux_moment_fs, ctx, runtime,"q2grpm");
  SnapArray qtot(simulation_is, spatial_ip, moment_fs, ctx, runtime, "qtot");

  SnapArray mat(simulation_is, spatial_ip, mat_fs, ctx, runtime, "mat");
  SnapArray sigt(material_is, IndexPartition::NO_PART, group_fs, 
                 ctx, runtime, "sigt");
  SnapArray siga(material_is, IndexPartition::NO_PART, group_fs,
                 ctx, runtime, "siga");
  SnapArray sigs(material_is, IndexPartition::NO_PART, group_fs,
                 ctx, runtime, "sigs");
  SnapArray slgg(slgg_is, IndexPartition::NO_PART, moment_fs, 
                 ctx, runtime, "slgg");

  SnapArray t_xs(simulation_is, spatial_ip, group_fs, ctx, runtime, "t_xs");
  SnapArray a_xs(simulation_is, spatial_ip, group_fs, ctx, runtime, "a_xs");
  SnapArray s_xs(simulation_is, spatial_ip, moment_fs, ctx, runtime, "s_xs");

  SnapArray vel(point_is, IndexPartition::NO_PART, group_fs, 
                ctx, runtime, "vel");
  SnapArray vdelt(point_is, IndexPartition::NO_PART, group_fs, 
                  ctx, runtime, "vdelt");
  SnapArray dinv(simulation_is, spatial_ip, angle_fs, 
                 ctx, runtime, "dinv"); 

  SnapArray *time_flux_even[8];
  SnapArray *time_flux_odd[8]; 
  for (int i = 0; i < 8; i++) {
    char name_buffer[64];
    snprintf(name_buffer, 63, "time flux even %d", i);
    time_flux_even[i] = new SnapArray(simulation_is, spatial_ip, angle_fs, 
                                      ctx, runtime, name_buffer);
    snprintf(name_buffer, 63, "time flux odd %d", i);
    time_flux_odd[i] = new SnapArray(simulation_is, spatial_ip, angle_fs,
                                     ctx, runtime, name_buffer);
  }

  // Initialize our data
  flux0.initialize();
  flux0po.initialize();
  flux0pi.initialize();
  if (num_moments > 1)
    fluxm.initialize();

  qi.initialize();
  q2grp0.initialize();
  if (num_moments > 1)
    q2grpm.initialize();
  qtot.initialize();

  mat.initialize<int>(1);
  sigt.initialize();
  siga.initialize();
  sigs.initialize();
  slgg.initialize();

  t_xs.initialize();
  a_xs.initialize();
  s_xs.initialize();

  vel.initialize();
  vdelt.initialize();
  dinv.initialize();

  for (int i = 0; i < 8; i++) {
    time_flux_even[i]->initialize();
    time_flux_odd[i]->initialize();
  }

  // Launch some tasks to initialize the application data
  if (material_layout != HOMOGENEOUS_LAYOUT)
  {
    InitMaterial init_material(*this, mat);
    init_material.dispatch(ctx, runtime);
  }
  if (source_layout != EVERYWHERE_SOURCE)
  {
    InitSource init_source(*this, qi);
    init_source.dispatch(ctx, runtime);
  }
  initialize_scattering(sigt, siga, sigs, slgg);
  initialize_velocity(vel, vdelt);

  // Tunables should be ready by now
  const unsigned outer_runahead = 
    outer_runahead_future.get_result<unsigned>(true/*silence warnings*/);
  assert(outer_runahead > 0);
  const unsigned inner_runahead = 
    inner_runahead_future.get_result<unsigned>(true/*silence warnings*/);
  assert(inner_runahead > 0);
  // Loop over time steps
  std::deque<Future> outer_converged_tests;
  std::deque<Future> inner_converged_tests;
  // Use this for when predicates evaluate to false, tasks can then
  // return true to indicate convergence
  const Future true_future = Future::from_value<bool>(runtime, true);
  // Iterate over time steps
  bool even_time_step = false;
  for (int cy = 0; cy < num_steps; ++cy)
  {
    even_time_step = !even_time_step;
    // Some of this is a little weird, you can in theory lift some
    // of this out the time stepping loop because the mock velocity 
    // array and the material array aren't changing, but I think that 
    // is just an artifact off SNAP and not a more general property of PARTISN, 
    // SNAP developers have now confirmed this so we'll leave this
    // here to be consistent with the original implementation of SNAP
    for (int g = 0; g < num_groups; g++)
    {
      ExpandCrossSection expxs(*this, siga, mat, a_xs, g);
      expxs.dispatch(ctx, runtime);
    }
    for (int g = 0; g < num_groups; g++)
    {
      ExpandCrossSection expxs(*this, sigt, mat, t_xs, g);
      expxs.dispatch(ctx, runtime);
    }
    for (int g = 0; g < num_groups; g++)
    {
      ExpandScatteringCrossSection expxs(*this, slgg, mat, s_xs, g);
      expxs.dispatch(ctx, runtime);
    }
    for (int g = 0; g < num_groups; g++)
    {
      CalculateGeometryParam geom(*this, t_xs, vdelt, dinv, g);
      geom.dispatch(ctx, runtime);
    }
    outer_converged_tests.clear();
    Predicate outer_pred = Predicate::TRUE_PRED;
    // The outer solve loop    
    for (int otno = 0; otno < max_outer_iters; ++otno)
    {
      // Do the outer source calculation 
      CalcOuterSource outer_src(*this, outer_pred, qi, slgg, mat, 
                                q2grp0, q2grpm, flux0, fluxm);
      outer_src.dispatch(ctx, runtime);
      // Save the fluxes
      save_fluxes(outer_pred, flux0, flux0po);
      // Do the inner solve
      inner_converged_tests.clear();
      Predicate inner_pred = Predicate::TRUE_PRED;
      Future inner_converged;
      // The inner solve loop
      for (int inno=0; inno < max_inner_iters; ++inno)
      {
        // Do the inner source calculation
        CalcInnerSource inner_src(*this, inner_pred, s_xs, flux0, fluxm,
                                  q2grp0, q2grpm, qtot);
        inner_src.dispatch(ctx, runtime);
        // Save the fluxes
        save_fluxes(inner_pred, flux0, flux0pi);
        flux0.initialize();
        // Perform the sweeps
        perform_sweeps(inner_pred, flux0, fluxm, qtot, vdelt, dinv, t_xs,
                       even_time_step ? time_flux_even : time_flux_odd,
                       even_time_step ? time_flux_odd : time_flux_even); 
        // Test for inner convergence
        TestInnerConvergence inner_conv(*this, inner_pred, 
                                        flux0, flux0pi, true_future);
        inner_converged = inner_conv.dispatch<AndReduction>(ctx, runtime);
        inner_converged_tests.push_back(inner_converged);
        // Update the next predicate
        Predicate converged = runtime->create_predicate(ctx, inner_converged);
        inner_pred = runtime->predicate_not(ctx, converged);
        // See if we've run far enough ahead
        if (inner_converged_tests.size() == inner_runahead)
        {
          Future f = inner_converged_tests.front();
          inner_converged_tests.pop_front();
          if (f.get_result<bool>(true/*silence warnings*/))
            break;
        }
      }
      // Test for outer convergence
      // SNAP says to skip this on the first iteration
      if (otno == 0)
        continue;
      TestOuterConvergence outer_conv(*this, outer_pred, flux0, flux0po, 
                                      inner_converged, true_future);
      Future outer_converged = 
        outer_conv.dispatch<AndReduction>(ctx, runtime);
      outer_converged_tests.push_back(outer_converged);
      // Update the next predicate
      Predicate converged = runtime->create_predicate(ctx, outer_converged);
      outer_pred = runtime->predicate_not(ctx, converged);
      // See if we've run far enough ahead
      if (outer_converged_tests.size() == outer_runahead)
      {
        Future f = outer_converged_tests.front();
        outer_converged_tests.pop_front();
        if (f.get_result<bool>(true/*silence warnings*/))
          break;
      }
    }
  }
  for (int i = 0; i < 8; i++) {
    delete time_flux_even[i];
    delete time_flux_odd[i];
  }
}

//------------------------------------------------------------------------------
void Snap::output(void)
//------------------------------------------------------------------------------
{
}

//------------------------------------------------------------------------------
void Snap::initialize_scattering(const SnapArray &sigt, const SnapArray &siga,
                             const SnapArray &sigs, const SnapArray &slgg) const
//------------------------------------------------------------------------------
{
  PhysicalRegion sigt_region = sigt.map();
  PhysicalRegion siga_region = siga.map();
  PhysicalRegion sigs_region = sigs.map();
  PhysicalRegion slgg_region = slgg.map();
  sigt_region.wait_until_valid(true/*ignore warnings*/);
  siga_region.wait_until_valid(true/*ignore warnings*/);
  sigs_region.wait_until_valid(true/*ignore warnings*/);
  slgg_region.wait_until_valid(true/*ignore warnings*/);

  // Names reflect fortran numbering from original snap
  const DomainPoint one = DomainPoint::from_point<1>(Point<1>(0));
  const DomainPoint two = DomainPoint::from_point<1>(Point<1>(1));
  std::vector<RegionAccessor<AccessorType::Generic,double> > fa_sigt(num_groups);
  std::vector<RegionAccessor<AccessorType::Generic,double> > fa_siga(num_groups);
  std::vector<RegionAccessor<AccessorType::Generic,double> > fa_sigs(num_groups);
  for (unsigned g = 0; g < num_groups; g++)
  {
    fa_sigt[g] = sigt_region.get_field_accessor(
        SNAP_ENERGY_GROUP_FIELD(g)).typeify<double>();
    fa_siga[g] = siga_region.get_field_accessor(
        SNAP_ENERGY_GROUP_FIELD(g)).typeify<double>();
    fa_sigs[g] = sigs_region.get_field_accessor(
        SNAP_ENERGY_GROUP_FIELD(g)).typeify<double>();
  }

  fa_sigt[0].write(one, 1.0);
  fa_siga[0].write(one, 0.5);
  fa_sigs[0].write(one, 0.5);
  for (unsigned g = 1; g < num_groups; g++)
  {
    fa_sigt[g].write(one, 0.01  * fa_sigt[g-1].read(one));
    fa_siga[g].write(one, 0.005 * fa_siga[g-1].read(one));
    fa_sigs[g].write(one, 0.005 * fa_sigs[g-1].read(one));
  }

  if (material_layout != HOMOGENEOUS_LAYOUT) {
    fa_sigt[0].write(two, 2.0);
    fa_siga[0].write(two, 0.8);
    fa_sigs[0].write(two, 1.2);
    for (unsigned g = 1; g < num_groups; g++)
    {
      fa_sigt[g].write(two, 0.01  * fa_sigt[g-1].read(two));
      fa_siga[g].write(two, 0.005 * fa_siga[g-1].read(two));
      fa_sigs[g].write(two, 0.005 * fa_sigs[g-1].read(two));
    }
  }

  std::vector<RegionAccessor<AccessorType::Generic,MomentQuad> > fa_slgg(num_groups); 
  for (unsigned g = 0; g < num_groups; g++)
    fa_slgg[g] = slgg_region.get_field_accessor(
        SNAP_ENERGY_GROUP_FIELD(g)).typeify<MomentQuad>();

  Point<2> p2 = Point<2>::ZEROES();
  if (num_groups == 1) {
    MomentQuad local;
    local[0] = fa_sigs[0].read(one);
    fa_slgg[0].write(DomainPoint::from_point<2>(p2), local); 
    if (material_layout != HOMOGENEOUS_LAYOUT) {
      p2.x[1] = 1; 
      local[0] = fa_sigs[0].read(two);
      fa_slgg[0].write(DomainPoint::from_point<2>(p2), local);
    }
  } else {
    MomentQuad local;
    for (unsigned g = 0; g < num_groups; g++) {
      p2.x[1] = g; 
      const DomainPoint dp = DomainPoint::from_point<2>(p2);
      local[0] = 0.2 * fa_sigs[g].read(one);
      fa_slgg[g].write(dp, local);
      if (g > 0) {
        const double t = 1.0 / double(g);
        for (unsigned g2 = 0; g2 < g; g2++) {
          local[0] = 0.1 * fa_sigs[g].read(one) * t;
          fa_slgg[g2].write(dp, local);
        }
      } else {
        local[0] = 0.3 * fa_sigs[g].read(one);
        fa_slgg[g].write(dp, local); 
      }

      if (g < (num_groups-1)) {
        const double t = 1.0 / double(num_groups-(g+1));
        for (unsigned g2 = g+1; g2 < num_groups; g2++) {
          local[0] = 0.7 * fa_sigs[g].read(one) * t;
          fa_slgg[g2].write(dp, local);
        }
      } else {
        local[0] = 0.9 * fa_sigs[g].read(one);
        fa_slgg[g].write(dp, local);
      }
    }
    if (material_layout != HOMOGENEOUS_LAYOUT) {
      p2.x[0] = 1;
      for (unsigned g = 0; g < num_groups; g++) {
        p2.x[1] = g; 
        const DomainPoint dp = DomainPoint::from_point<2>(p2);
        local[0] = 0.5 * fa_sigs[g].read(two);
        fa_slgg[g].write(dp, local);
        if (g > 0) {
          const double t = 1.0 / double(g);
          for (unsigned g2 = 0; g2 < g; g2++) {
            local[0] = 0.1 * fa_sigs[g].read(two) * t;
            fa_slgg[g2].write(dp, local);
          }
        } else {
          local[0] = 0.6 * fa_sigs[g].read(two);
          fa_slgg[g].write(dp, local); 
        }

        if (g < (num_groups-1)) {
          const double t = 1.0 / double(num_groups-(g+1));
          for (unsigned g2 = g+1; g2 < num_groups; g2++) {
            local[0] = 0.4 * fa_sigs[g].read(two) * t;
            fa_slgg[g2].write(dp, local);
          }
        } else {
          local[0] = 0.9 * fa_sigs[g].read(two);
          fa_slgg[g].write(dp, local);
        }
      }
    }
  }
  if (num_moments > 1) 
  {
    p2 = Point<2>::ZEROES();
    for (int m = 1; m < num_moments; m++) {
      for (int g = 0; g < num_groups; g++) {
        p2.x[1] = g;
        DomainPoint dp = DomainPoint::from_point<2>(p2);
        for (int g2 = 0; g2 < num_groups; g2++) {
          MomentQuad quad = fa_slgg[g2].read(dp);
          quad[m] = ((m == 1) ? 0.1 : 0.5) * quad[m-1];
          fa_slgg[g2].write(dp, quad);
        }
      }
    }
    if (material_layout != HOMOGENEOUS_LAYOUT) {
      p2.x[0] = 1;
      for (int m = 1; m < num_moments; m++) {
        for (int g = 0; g < num_groups; g++) {
          p2.x[1] = g;
          DomainPoint dp = DomainPoint::from_point<2>(p2);
          for (int g2 = 0; g2 < num_groups; g2++) {
            MomentQuad quad = fa_slgg[g2].read(dp);
            quad[m] = ((m == 1) ? 0.8 : 0.6) * quad[m-1];
            fa_slgg[g2].write(dp, quad);
          }
        }
      }
    }
  }

  sigt.unmap(sigt_region);
  siga.unmap(siga_region);
  sigs.unmap(sigs_region);
  slgg.unmap(slgg_region);
}

//------------------------------------------------------------------------------
void Snap::initialize_velocity(const SnapArray &vel, 
                               const SnapArray &vdelt) const
//------------------------------------------------------------------------------
{
  PhysicalRegion vel_region = vel.map();
  PhysicalRegion vdelt_region = vdelt.map();
  vel_region.wait_until_valid(true/*ignore warnings*/);
  vdelt_region.wait_until_valid(true/*ignore warnings*/);
  const DomainPoint dp = DomainPoint::from_point<1>(Point<1>(0));
  for (int g = 0; g < num_groups; g++) 
  {
    RegionAccessor<AccessorType::Generic,double> fa_vel = 
      vel_region.get_field_accessor(SNAP_ENERGY_GROUP_FIELD(g)).typeify<double>();
    RegionAccessor<AccessorType::Generic,double> fa_vdelt = 
      vdelt_region.get_field_accessor(SNAP_ENERGY_GROUP_FIELD(g)).typeify<double>();
    const double v = double(Snap::num_groups - g);
    fa_vel.write(dp, v);
    if (Snap::time_dependent)
      fa_vdelt.write(dp, 2.0 / (Snap::dt * v));
    else
      fa_vdelt.write(dp, 0.0);
  }
  vel.unmap(vel_region);
  vdelt.unmap(vdelt_region);
}

//------------------------------------------------------------------------------
void Snap::save_fluxes(const Predicate &pred, 
                       const SnapArray &src, const SnapArray &dst) const
//------------------------------------------------------------------------------
{
  // Build the CopyLauncher
  CopyLauncher launcher(pred);
  launcher.add_copy_requirements(
      RegionRequirement(LogicalRegion::NO_REGION, READ_ONLY, 
                        EXCLUSIVE, src.get_region()),
      RegionRequirement(LogicalRegion::NO_REGION, WRITE_DISCARD,
                        EXCLUSIVE, dst.get_region()));
  const std::set<FieldID> &src_fields = src.get_regular_fields();
  RegionRequirement &src_req = launcher.src_requirements.back();
  src_req.privilege_fields = src_fields;
  src_req.instance_fields.insert(src_req.instance_fields.end(),
      src_fields.begin(), src_fields.end());
  const std::set<FieldID> &dst_fields = dst.get_regular_fields();
  RegionRequirement &dst_req = launcher.dst_requirements.back();
  dst_req.privilege_fields = dst_fields;
  dst_req.instance_fields.insert(dst_req.instance_fields.end(),
      dst_fields.begin(), dst_fields.end());
  // Iterate over the sub-regions and issue copies for each separately
  for (GenericPointInRectIterator<3> color_it(launch_bounds); 
        color_it; color_it++)   
  {
    DomainPoint dp = DomainPoint::from_point<3>(color_it.p);
    src_req.region = src.get_subregion(dp);
    dst_req.region = dst.get_subregion(dp);
    runtime->issue_copy_operation(ctx, launcher);
  }
}

//------------------------------------------------------------------------------
void Snap::perform_sweeps(const Predicate &pred, const SnapArray &flux,
                          const SnapArray &fluxm, const SnapArray &qtot, 
                          const SnapArray &vdelt, const SnapArray &dinv, 
                          const SnapArray &t_xs, SnapArray *time_flux_in[8],
                          SnapArray *time_flux_out[8]) const
//------------------------------------------------------------------------------
{
  // Loop over the corners
  for (int corner = 0; corner < num_corners; corner++)
  {
    // Compute the projection functions for this corner
    int ghost_offsets[3] = { 0, 0, 0 };
    for (int i = 0; i < num_dims; i++)
      ghost_offsets[i] = (corner & (0x1 << i)) >> i;
    const std::vector<Domain> &launch_domains = wavefront_domains[corner];
    // Then loop over the energy groups
    for (int group = 0; group < num_groups; group++)
    {
      // Launch the sweep from this corner for the given field
      // We alternate between even and odd ghost fields since
      // Legion can't prove that some of the region requirements
      // are non-interfering with it's current projection analysis
      MiniKBATask mini_kba_even(*this, pred, true/*even*/, flux, fluxm, 
                                qtot, vdelt, dinv, t_xs, 
                                *time_flux_in[corner], *time_flux_out[corner],
                                group, corner, ghost_offsets);
      MiniKBATask mini_kba_odd(*this, pred, false/*even*/, flux, fluxm,
                                qtot, vdelt, dinv, t_xs,
                                *time_flux_in[corner], *time_flux_out[corner],
                                group, corner, ghost_offsets);
      bool even = true;
      for (unsigned idx = 0; idx < launch_domains.size(); idx++)
      {
        if (even)
          mini_kba_even.dispatch_wavefront(idx, launch_domains[idx], 
                                           ctx, runtime);
        else
          mini_kba_odd.dispatch_wavefront(idx, launch_domains[idx],
                                          ctx, runtime);
        even = !even;
      }
    }
  }
}

//------------------------------------------------------------------------------
/*static*/ void Snap::snap_top_level_task(const Task *task,
                                     const std::vector<PhysicalRegion> &regions,
                                     Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
  printf("Welcome to Legion-SNAP!\n");
  report_arguments();
  Snap snap(ctx, runtime); 
  snap.setup();
  snap.transport_solve();
  snap.output();
}

static void skip_line(FILE *f)
{
  char buffer[80];
  assert(fgets(buffer, 79, f) > 0);
}

static void read_int(FILE *f, const char *name, int &result)
{
  char format[80];
  sprintf(format,"  %s=%%d", name);
  assert(fscanf(f, format, &result) > 0);
}

static void read_bool(FILE *f, const char *name, bool &result)
{
  char format[80];
  sprintf(format,"  %s=%%d", name);
  int temp = 0;
  assert(fscanf(f, format, &temp) > 0);
  result = (temp != 0);
}

static void read_double(FILE *f, const char *name, double &result)
{
  char format[80];
  sprintf(format,"  %s=%%lf", name);
  assert(fscanf(f, format, &result) > 0);
}

int Snap::num_dims = 1;
int Snap::nx_chunks = 4;
int Snap::ny_chunks = 1;
int Snap::nz_chunks = 1;
int Snap::nx = 4;
double Snap::lx = 1.0;
int Snap::ny = 1;
double Snap::ly = 1.0;
int Snap::nz = 1;
double Snap::lz = 1.0;
int Snap::num_moments = 1;
int Snap::num_angles = 1;
int Snap::num_groups = 1;
double Snap::convergence_eps = 1e-4;
int Snap::max_inner_iters = 5;
int Snap::max_outer_iters = 100;
bool Snap::time_dependent = false;
double Snap::total_sim_time = 0.0;
int Snap::num_steps = 1;
Snap::MaterialLayout Snap::material_layout = HOMOGENEOUS_LAYOUT;
Snap::SourceLayout Snap::source_layout = EVERYWHERE_SOURCE; 
bool Snap::dump_scatter = false;
bool Snap::dump_iteration = false;
int Snap::dump_flux = 0;
bool Snap::flux_fixup = false;
bool Snap::dump_solution = false;
int Snap::dump_kplane = 0;
int Snap::dump_population = 0;
bool Snap::minikba_sweep = true;
bool Snap::single_angle_copy = true;

int Snap::num_corners = 1;
std::vector<std::vector<DomainPoint> > Snap::wavefront_map[8];
std::vector<std::vector<Point<3> > > Snap::chunk_wavefronts;
double Snap::dt;
int Snap::cmom;
int Snap::num_octants;
double Snap::hi;
double Snap::hj;
double Snap::hk;
double* Snap::mu;
double* Snap::w;
double *Snap::wmu;
double* Snap::eta;
double* Snap::weta;
double* Snap::xi;
double* Snap::wxi;
double* Snap::ec;
int Snap::lma[4];

//------------------------------------------------------------------------------
/*static*/ void Snap::parse_arguments(int argc, char **argv)
//------------------------------------------------------------------------------
{
  if (argc < 2) {
    printf("No input file specified. Exiting.\n");
    exit(1);
  }
  printf("Reading input file %s\n", argv[1]);
  FILE *f = fopen(argv[1], "r");
  int dummy_int = 0;
  skip_line(f);
  skip_line(f);
  read_int(f, "npey", ny_chunks);
  read_int(f, "npez", nz_chunks);
  read_int(f, "ichunk", nx_chunks);
  read_int(f, "nthreads", dummy_int);
  read_int(f, "nnested", dummy_int);
  read_int(f, "ndimen", num_dims);
  read_int(f, "nx", nx);
  read_double(f, "lx", lx);
  read_int(f, "ny", ny);
  read_double(f, "ly", ly);
  read_int(f, "nz", nz);
  read_double(f, "lz", lz);
  read_int(f, "nmom", num_moments);
  read_int(f, "nang", num_angles);
  read_int(f, "ng", num_groups);
  read_double(f, "epsi", convergence_eps);
  read_int(f, "iitm", max_inner_iters);
  read_int(f, "oitm", max_outer_iters);
  read_bool(f, "timedep", time_dependent);
  read_double(f, "tf", total_sim_time);
  read_int(f, "nsteps", num_steps);
  int mat_opt = 0;
  read_int(f, "mat_opt", mat_opt);
  switch (mat_opt)
  {
    case 0:
      {
        material_layout = HOMOGENEOUS_LAYOUT;
        break;
      }
    case 1:
      {
        material_layout = CENTER_LAYOUT;
        break;
      }
    case 2:
      {
        material_layout = CORNER_LAYOUT;
        break;
      }
    default:
      assert(false);
  }
  int src_opt = 0;
  read_int(f, "src_opt", src_opt);
  switch (src_opt)
  {
    case 0:
      {
        source_layout = EVERYWHERE_SOURCE;
        break;
      }
    case 1:
      {
        source_layout = CENTER_SOURCE;
        break;
      }
    case 2:
      {
        source_layout = CORNER_SOURCE;
        break;
      }
    case 3:
      {
        source_layout = MMS_SOURCE;
        break;
      }
    default:
      assert(false);
  }
  read_bool(f, "scatp", dump_scatter);
  read_bool(f, "it_det", dump_iteration);
  read_int(f, "fluxp", dump_flux);
  read_bool(f, "fixup", flux_fixup);
  read_bool(f, "soloutp", dump_solution);
  read_int(f, "kplane", dump_kplane);
  read_int(f, "popout", dump_population);
  read_bool(f, "swp_typ", minikba_sweep);
  read_bool(f, "angcpy", single_angle_copy);
  if (single_angle_copy)
  {
    printf("ERROR: angcpy=1 is not currently supported.\n");
    printf("Legion SNAP currently requires two copies of angle flux\n");
    exit(1);
  }
  fclose(f);
  if (!minikba_sweep)
  {
    printf("ERROR: swp_type=0 is not currently supported.\n");
    printf("Legion SNAP currently implements only mini-kba sweeps\n");
    exit(1);
  }
  // Check all the conditions
  assert((1 <= num_dims) && (num_dims <= 3));
  assert((1 <= nx_chunks) && (nx_chunks <= nx));
  assert((1 <= ny_chunks) && (ny_chunks <= ny));
  assert((1 <= nz_chunks) && (nz_chunks <= nz));
  assert(4 <= nx);
  assert(0.0 < lx);
  assert(4 <= ny);
  assert(0.0 < ly);
  assert(4 <= ny);
  assert(0.0 < lz);
  assert((nx % nx_chunks) == 0);
  assert((ny % ny_chunks) == 0);
  assert((nz % nz_chunks) == 0);
  assert((1 <= num_moments) && (num_moments <= 4));
  assert(1 <= num_angles);
  assert(1 <= num_groups);
  assert((0.0 < convergence_eps) && (convergence_eps < 1e-2));
  assert(1 <= max_inner_iters);
  assert(1 <= max_outer_iters);
  if (time_dependent)
    assert(0.0 <= total_sim_time);
  assert(1 <= num_steps);
  // Some derived quantities
  for (int i = 0; i < num_dims; i++)
    num_corners *= 2;
  compute_wavefronts();
  compute_derived_globals();
}

static bool contains_point(Point<3> &point, int xlo, int xhi, 
                           int ylo, int yhi, int zlo, int zhi)
{
  if ((point[0] < xlo) || (point[0] > xhi))
    return false;
  if ((point[1] < ylo) || (point[1] > yhi))
    return false;
  if ((point[2] < zlo) || (point[2] > zhi))
    return false;
  return true;
}

//------------------------------------------------------------------------------
/*static*/ void Snap::compute_wavefronts(void)
//------------------------------------------------------------------------------
{
  const int chunks[3] = { nx_chunks, ny_chunks, nz_chunks };
  // Compute the mapping from corners to wavefronts
  for (int corner = 0; corner < num_corners; corner++)
  {
    Point<3> strides[3] = 
      { Point<3>::ZEROES(), Point<3>::ZEROES(), Point<3>::ZEROES() };
    Point<3> start;
    for (int i = 0; i < num_dims; i++)
    {
      start.x[i] = ((corner & (0x1 << i)) ? chunks[i] : 1);
      strides[i].x[i] = ((corner & (0x1 << i)) ? -1 : 1);
    }
    std::set<DomainPoint> current_points;
    current_points.insert(DomainPoint::from_point<3>(Point<3>(start)));
    // Do a little BFS to handle weird rectangle shapes correctly
    unsigned wavefront_number = 0;
    while (!current_points.empty())
    {
      wavefront_map[corner].push_back(std::vector<DomainPoint>());
      std::vector<DomainPoint> &wavefront_points = 
                          wavefront_map[corner][wavefront_number];
      std::set<DomainPoint> next_points;
      for (std::set<DomainPoint>::const_iterator it = current_points.begin();
            it != current_points.end(); it++)
      {
        // Save the point in this wavefront
        wavefront_points.push_back(*it);
        Point<3> point = it->get_point<3>();
        for (int i = 0; i < num_dims; i++)
        {
          Point<3> next = point + strides[i];
          if (contains_point(next, 1, nx_chunks, 1, ny_chunks, 1, nz_chunks))
            next_points.insert(DomainPoint::from_point<3>(next));
        }
      }
      current_points = next_points;
      wavefront_number++;
    }
  }
  // Now compute the chunk wavefronts
  // Total number of wavefronts is nx + ny + nz - 2
  const int total_wavefronts = nx + ny + nz - 2;
  chunk_wavefronts.resize(total_wavefronts);
  int current_point[3];
  for (current_point[0] = 0; current_point[0] < nx; current_point[0]++) 
  {
    for (current_point[1] = 0; current_point[1] < ny; current_point[1]++)
    {
      for (current_point[2] = 0; current_point[2] < nz; current_point[2]++)
      {
        const int wavefront = 
          current_point[0] + current_point[1] + current_point[2];
        assert(wavefront < total_wavefronts);
        chunk_wavefronts[wavefront].push_back(Point<3>(current_point)); 
      }
    }
  }
}

//------------------------------------------------------------------------------
/*static*/ void Snap::compute_derived_globals(void)
//------------------------------------------------------------------------------
{
  dt = total_sim_time / double(num_steps);

  cmom = num_moments;
  num_octants = 2;
  hi = 2.0 / (lx / double(nx));
  hj = 2.0 / (ly / double(ny));
  hk = 2.0 / (lz / double(nz));
  const size_t buffer_size = num_angles * sizeof(double);
  mu = (double*)malloc(buffer_size);
  w = (double*)malloc(buffer_size);
  wmu = (double*)malloc(buffer_size);
  eta = (double*)malloc(buffer_size);
  weta = (double*)malloc(buffer_size);
  xi = (double*)malloc(buffer_size);
  wxi = (double*)malloc(buffer_size);

  memset(mu, 0, buffer_size);
  memset(w, 0, buffer_size);
  memset(wmu, 0, buffer_size);
  memset(eta, 0, buffer_size);
  memset(weta, 0, buffer_size);
  memset(xi, 0, buffer_size);
  memset(wxi, 0, buffer_size);

  if (num_dims > 1) {
    cmom = num_moments * (num_moments+1) / 2;
    num_octants = 4;
  }

  if (num_dims > 2) { 
    cmom = num_moments * num_moments;
    num_octants = 8;
  }

  const double dm = 1.0 / double(num_angles);

  mu[0] = 0.5 * dm;
  for (int i = 1; i < num_angles; i++)
    mu[i] = mu[i-1] + dm;
  if (num_dims > 1) { 
    eta[0] = 1.0 - 0.5 * dm;
    for (int i = 1; i < num_angles; i++)
      eta[i] = eta[i-1] - dm;

    if (num_dims > 2) {
      for (int i = 0; i < num_angles; i++) {
        const double t = mu[i]*mu[i] + eta[i]*eta[i];
        xi[i] = sqrt( 1.0 - t );
      }
    }
  }

  if (num_dims == 1) {
    for (int i = 0; i < num_angles; i++)
      w[i] = 0.5 / double(num_angles);
  } else if (num_dims == 2) {
    for (int i = 0; i < num_angles; i++)
      w[i] = 0.25 / double(num_angles);
  } else if (num_dims == 3) {
    for (int i = 0; i < num_angles; i++)
      w[i] = 0.125 / double(num_angles);
  } else 
    assert(false);

  const size_t ec_size = num_angles * cmom * num_octants * sizeof(double);
  ec = (double*)malloc(ec_size);
  memset(ec, 0, ec_size);

  for (int i = 0; i < 4; i++)
    lma[i] = 0;

  switch (num_dims)
  {
    case 1:
      {
        for (int i = 0; i < num_angles; i++)
          wmu[i] = w[i] * mu[i];
        for (int i = 0; i < 4; i++)
          lma[i] = 1;
        for (int id = 0; id < 2; id++) {
          int is = -1;
          if (id == 1) 
            is = 1;
          for (int idx = 0; idx < num_angles; idx++)
            ec[id * num_angles * num_moments + idx] = 1.0;
          for (int l = 1; l < num_moments; l++)
            for (int idx = 0; idx < num_angles; idx++)
              ec[id * num_angles * num_moments + l * num_angles + idx] = 
                pow(is*mu[idx], double(2*(l+1)-3)); 
        }
        break;
      }
    case 2:
      {
        for (int i = 0; i < num_angles; i++)
          wmu[i] = w[i] * mu[i];
        for (int i = 0; i < num_angles; i++)
          weta[i] = w[i] * eta[i];
        for (int l = 0; l < num_moments; l++)
          lma[l] = l+1;
        for (int jd = 0; jd < 2; jd++) {
          int js = -1;
          if (jd == 1)
            js = 1;
          for (int id = 0; id < 2; id++) {
            int is = -1;
            if (id == 1)
              is = 1;
            int oct = 2*jd + id;
            for (int idx = 0; idx < num_angles; idx++)
              ec[oct * num_angles * num_moments + idx] = 1.0;
            int moment = 1;
            for (int l = 1; l < num_moments; l++) {
              for (int m = 0; m < l; m++) {
                for (int idx = 0; idx < num_angles; idx++)
                  ec[oct * num_angles * num_moments + moment * num_angles + idx] = 
                    pow(is*mu[idx],2*(l+1)-3) * pow(js*eta[idx],m);
                moment++;
              }
            }
          }
        }
        break;
      }
    case 3:
      {
        for (int i = 0; i < num_angles; i++)
          wmu[i] = w[i] * mu[i];
        for (int i = 0; i < num_angles; i++)
          weta[i] = w[i] * eta[i];
        for (int i = 0; i < num_angles; i++)
          wxi[i] = w[i] * xi[i];

        for (int l = 0; l < num_moments; l++)
          lma[l] = 2*(l+1) - 1;

        for (int kd = 0; kd < 2; kd++) {
          int ks = -1;
          if (kd == 1)
            ks = 1;
          for (int jd = 0; jd < 2; jd++) {
            int js = -1;
            if (jd == 1)
              js = 1;
            for (int id = 0; id < 2; id++) {
              int is = -1;
              if (id == 1)
                is = 1;
              int oct = 4 * kd + 2 * jd + id;
              for (int idx = 0; idx < num_angles; idx++)
                ec[oct * num_angles * num_moments + idx] = 1.0;
              int moment = 1;
              for (int l = 1; l < num_moments; l++) {
                for (int m = 0; m < (2*(l+1)-1); m++) {
                  for (int idx = 0; idx < num_angles; idx++)
                    ec[oct * num_angles * num_moments + moment * num_angles + idx] = 
                      pow(is*mu[idx], 2*(l+1)-3) * pow(ks * xi[idx] * js * eta[idx], m);
                  moment++;
                }
              }
            }
          }
        }
        break;
      }
    default:
      assert(false);
  }
}

//------------------------------------------------------------------------------
/*static*/ void Snap::report_arguments(void)
//------------------------------------------------------------------------------
{
  printf("Dimensions: %d\n", num_dims);
  printf("X-Chunks: %d\n", nx_chunks);
  printf("Y-Chunks: %d\n", ny_chunks);
  printf("Z-Chunks: %d\n", nz_chunks);
  printf("nx,ny,nz: %d,%d,%d\n", nx, ny, nz);
  printf("lx,ly,lz: %.8g,%.8g,%.8g\n", lx, ly, lz);
  printf("Moments: %d\n", num_moments);
  printf("Angles: %d\n", num_angles);
  printf("Groups: %d\n", num_groups);
  printf("Convergence: %.8g\n", convergence_eps);
  printf("Max Inner Iterations: %d\n", max_inner_iters);
  printf("Max Outer Iterations: %d\n", max_outer_iters);
  printf("Time Dependent: %s\n", (time_dependent ? "Yes" : "No"));
  printf("Total Simulation Time: %.8g\n", total_sim_time);
  printf("Total Steps: %d\n", num_steps);
  printf("Material Layout: %s\n", ((material_layout == HOMOGENEOUS_LAYOUT) ? 
        "Homogenous Layout" : (material_layout == CENTER_LAYOUT) ? 
        "Center Layout" : "Corner Layout"));
  printf("Source Layout: %s\n", ((source_layout == EVERYWHERE_SOURCE) ? 
        "Everywhere" : (source_layout == CENTER_SOURCE) ? "Center" :
        (source_layout == CORNER_SOURCE) ? "Corner" : "MMS"));
  printf("Dump Scatter: %s\n", dump_scatter ? "Yes" : "No");
  printf("Dump Iteration: %s\n", dump_iteration ? "Yes" : "No");
  printf("Dump Flux: %s\n", (dump_flux == 2) ? "All" : 
      (dump_flux == 1) ? "Scalar" : "No");
  printf("Fixup Flux: %s\n", flux_fixup ? "Yes" : "No");
  printf("Dump Solution: %s\n", dump_solution ? "Yes" : "No");
  printf("Dump Kplane: %d\n", dump_kplane);
  printf("Dump Population: %s\n", (dump_population == 2) ? "All" : 
      (dump_population == 1) ? "Final" : "No");
  printf("Mini-KBA Sweep: %s\n", minikba_sweep ? "Yes" : "No");
  printf("Single Angle Copy: %s\n", single_angle_copy ? "Yes" : "No");
}

//------------------------------------------------------------------------------
/*static*/ void Snap::perform_registrations(void)
//------------------------------------------------------------------------------
{
  TaskVariantRegistrar registrar(SNAP_TOP_LEVEL_TASK_ID, "snap_main_variant");
  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  Runtime::preregister_task_variant<snap_top_level_task>(registrar,
                          Snap::task_names[SNAP_TOP_LEVEL_TASK_ID]);
  Runtime::set_top_level_task_id(SNAP_TOP_LEVEL_TASK_ID);
  Runtime::set_registration_callback(mapper_registration);
  // Now register all the task variants
  InitMaterial::preregister_cpu_variants();
  InitSource::preregister_cpu_variants();
  CalcOuterSource::preregister_all_variants();
  TestOuterConvergence::preregister_all_variants();
  CalcInnerSource::preregister_all_variants();
  TestInnerConvergence::preregister_all_variants();
  MiniKBATask::preregister_all_variants();
  ExpandCrossSection::preregister_all_variants();
  ExpandScatteringCrossSection::preregister_all_variants();
  CalculateGeometryParam::preregister_all_variants();
  // Register projection functors
  Runtime::preregister_projection_functor(SWEEP_PROJECTION, 
                        new SnapSweepProjectionFunctor());
  for (int dim = 0; dim < num_dims; dim++)
    for (int offset = 0; offset < 2; offset++)
    {
      SnapProjectionID ghost_id = SNAP_GHOST_PROJECTION(dim, offset);
      Runtime::preregister_projection_functor(ghost_id,
          new SnapGhostProjectionFunctor(dim, offset));
    }
  // Finally register our reduction operators
  Runtime::register_reduction_op<AndReduction>(AndReduction::REDOP_ID);
  Runtime::register_reduction_op<SumReduction>(SumReduction::REDOP_ID);
  Runtime::register_reduction_op<QuadReduction>(QuadReduction::REDOP_ID);
}

//------------------------------------------------------------------------------
/*static*/ void Snap::mapper_registration(Machine machine, Runtime *runtime,
                                         const std::set<Processor> &local_procs)
//------------------------------------------------------------------------------
{
  MapperRuntime *mapper_rt = runtime->get_mapper_runtime();
  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    runtime->replace_default_mapper(new SnapMapper(mapper_rt, machine, 
                                                   *it, "SNAP Mapper"), *it);
  }
}

//------------------------------------------------------------------------------
Snap::SnapMapper::SnapMapper(MapperRuntime *rt, Machine machine, 
                             Processor local, const char *mapper_name)
  : DefaultMapper(rt, machine, local, mapper_name)
//------------------------------------------------------------------------------
{
}

//------------------------------------------------------------------------------
void Snap::SnapMapper::select_tunable_value(const MapperContext ctx,
                                            const Task& task,
                                            const SelectTunableInput& input,
                                                  SelectTunableOutput& output)
//------------------------------------------------------------------------------
{
  switch (input.tunable_id)
  {
    case OUTER_RUNAHEAD_TUNABLE:
      {
        runtime->pack_tunable<unsigned>(4, output);
        break;
      }
    case INNER_RUNAHEAD_TUNABLE:
      {
        runtime->pack_tunable<unsigned>(8, output);
        break;
      }
    default:
      // Fall back to the default mapper
      DefaultMapper::select_tunable_value(ctx, task, input, output);
  }
}

//------------------------------------------------------------------------------
SnapArray::SnapArray(IndexSpace is, IndexPartition ip, FieldSpace fs,
                     Context c, Runtime *rt, const char *name)
  : ctx(c), runtime(rt)
//------------------------------------------------------------------------------
{
  char name_buffer[64];
  snprintf(name_buffer,63,"%s Logical Region", name);
  lr = runtime->create_logical_region(ctx, is, fs);
  runtime->attach_name(lr, name_buffer);
  if (ip.exists())
  {
    lp = runtime->get_logical_partition(lr, ip);
    snprintf(name_buffer,63,"%s Spatial Partition", name);
    runtime->attach_name(lp, name_buffer);
  }
  std::vector<FieldID> all_fields;
  runtime->get_field_space_fields(fs, all_fields);
  for (std::vector<FieldID>::const_iterator it = all_fields.begin();
        it != all_fields.end(); it++)
  {
    if ((*it) < Snap::FID_GROUP_MAX)
      regular_fields.insert(*it);
    else
      ghost_fields.insert(*it);
  }
}

//------------------------------------------------------------------------------
SnapArray::SnapArray(const SnapArray &rhs)
  : ctx(rhs.ctx), runtime(rhs.runtime)
//------------------------------------------------------------------------------
{
  // should never be called
  assert(false);
}

//------------------------------------------------------------------------------
SnapArray::~SnapArray(void)
//------------------------------------------------------------------------------
{
  runtime->destroy_logical_region(ctx, lr);
}

//------------------------------------------------------------------------------
SnapArray& SnapArray::operator=(const SnapArray &rhs)
//------------------------------------------------------------------------------
{
  // should never be called
  assert(false);
  return *this;
}

//------------------------------------------------------------------------------
LogicalRegion SnapArray::get_subregion(const DomainPoint &color) const
//------------------------------------------------------------------------------
{
  assert(lp.exists());
  // See if we already cached the result
  std::map<DomainPoint,LogicalRegion>::const_iterator finder = 
    subregions.find(color);
  if (finder != subregions.end())
    return finder->second;
  LogicalRegion result = runtime->get_logical_subregion_by_color(lp, color);
  // Save the result for later
  subregions[color] = result;
  return result;
}

//------------------------------------------------------------------------------
void SnapArray::initialize(void) const
//------------------------------------------------------------------------------
{
  assert(!regular_fields.empty());
  // Assume all the fields are the same size
  size_t field_size = runtime->get_field_size(lr.get_field_space(),
                                              *(regular_fields.begin()));
  void *buffer = malloc(field_size);
  memset(buffer, 0, field_size);
  FillLauncher launcher(lr, lr, TaskArgument(buffer, field_size));
  launcher.fields = regular_fields;
  runtime->fill_fields(ctx, launcher);
  free(buffer);
}

//------------------------------------------------------------------------------
template<typename T>
void SnapArray::initialize(T value) const
//------------------------------------------------------------------------------
{
  FillLauncher launcher(lr, lr, TaskArgument(&value, sizeof(value)));
  launcher.fields = regular_fields;
  runtime->fill_fields(ctx, launcher);
}

//------------------------------------------------------------------------------
PhysicalRegion SnapArray::map(void) const
//------------------------------------------------------------------------------
{
  InlineLauncher launcher(RegionRequirement(lr, READ_WRITE, EXCLUSIVE, lr));
  launcher.requirement.privilege_fields = regular_fields;
  return runtime->map_region(ctx, launcher);
}

//------------------------------------------------------------------------------
void SnapArray::unmap(const PhysicalRegion &region) const
//------------------------------------------------------------------------------
{
  runtime->unmap_region(ctx, region);
}

//------------------------------------------------------------------------------
SnapSweepProjectionFunctor::SnapSweepProjectionFunctor(void)
  : ProjectionFunctor()
//------------------------------------------------------------------------------
{
  // Set up the cache now so we don't need a lock later
  for (int corner = 0; corner < Snap::num_corners; corner++)
    for (int index = 0; index < MINI_KBA_NON_GHOST_REQUIREMENTS; index++)
    {
      cache[corner][index].resize(Snap::wavefront_map[corner].size());
      cache_valid[corner][index].resize(Snap::wavefront_map[corner].size());
      for (int wavefront = 0; 
            wavefront < int(Snap::wavefront_map[corner].size()); wavefront++)
      {
        cache[corner][index][wavefront].resize(
            Snap::wavefront_map[corner][wavefront].size(), 
            LogicalRegion::NO_REGION);
        cache_valid[corner][index][wavefront].resize(
            Snap::wavefront_map[corner][wavefront].size(), false/*valid*/);
      }
    }
}

//------------------------------------------------------------------------------
LogicalRegion SnapSweepProjectionFunctor::project(Context ctx, Task *task,
            unsigned index, LogicalRegion upper_bound, const DomainPoint &point)
//------------------------------------------------------------------------------
{
  // should never be called
  assert(false);
  return LogicalRegion::NO_REGION;
}

//------------------------------------------------------------------------------
LogicalRegion SnapSweepProjectionFunctor::project(Context ctx, Task *task,
         unsigned index, LogicalPartition upper_bound, const DomainPoint &point)
//------------------------------------------------------------------------------
{
  assert(task->task_id == Snap::MINI_KBA_TASK_ID);
  assert(index < MINI_KBA_NON_GHOST_REQUIREMENTS);
  assert(point.get_dim() == 1);
  Point<1> p = point.get_point<1>();
  // Figure out which wavefront and corner we are in
  unsigned wavefront = ((const MiniKBATask::MiniKBAArgs*)task->args)->wavefront;
  unsigned corner = ((const MiniKBATask::MiniKBAArgs*)task->args)->corner;
  assert(p[0] < int(cache_valid[corner][index][wavefront].size()));
  // Check to see if it is in the cache
  if (cache_valid[corner][index][wavefront][p[0]])
    return cache[corner][index][wavefront][p[0]];
  // Not valid, need to go get the result
  LogicalRegion result = runtime->get_logical_subregion_by_color(upper_bound,
                                Snap::wavefront_map[corner][wavefront][p[0]]);
  cache[corner][index][wavefront][p[0]] = result;
  cache_valid[corner][index][wavefront][p[0]] = true;
  return result;
}

//------------------------------------------------------------------------------
SnapGhostProjectionFunctor::SnapGhostProjectionFunctor(int d, int o)
  : ProjectionFunctor(), dim(d), offset(o),
    color((offset == 0) ? Snap::LO_GHOST : Snap::HI_GHOST), 
    stride(get_stride(d,o))
//------------------------------------------------------------------------------
{
  // Set up the cache now so we don't need a lock later
  for (int corner = 0; corner < Snap::num_corners; corner++)
  {
    cache[corner].resize(Snap::wavefront_map[corner].size());
    cache_valid[corner].resize(Snap::wavefront_map[corner].size());
    for (int wavefront = 0; 
          wavefront < int(Snap::wavefront_map[corner].size()); wavefront++)
    {
      cache[corner][wavefront].resize(
          Snap::wavefront_map[corner][wavefront].size(), 
          LogicalRegion::NO_REGION);
      cache_valid[corner][wavefront].resize(
          Snap::wavefront_map[corner][wavefront].size(), false/*valid*/);
    }
  }
}

//------------------------------------------------------------------------------
/*static*/ Point<3> SnapGhostProjectionFunctor::get_stride(int dim, int offset)
//------------------------------------------------------------------------------
{
  assert(dim < 3);
  int result[3] = { 0, 0, 0 };
  result[dim] = (offset == 0) ? -1 : 1;
  return Point<3>(result);
}

//------------------------------------------------------------------------------
LogicalRegion SnapGhostProjectionFunctor::project(Context ctx, Task *task,
            unsigned index, LogicalRegion upper_bound, const DomainPoint &point)
//------------------------------------------------------------------------------
{
  // should never be called
  assert(false);
  return LogicalRegion::NO_REGION;
}

//------------------------------------------------------------------------------
LogicalRegion SnapGhostProjectionFunctor::project(Context ctx, Task *task,
         unsigned index, LogicalPartition upper_bound, const DomainPoint &point)
//------------------------------------------------------------------------------
{
  assert(task->task_id == Snap::MINI_KBA_TASK_ID);
  assert(point.get_dim() == 1);
  Point<1> p = point.get_point<1>();
  // Figure out which wavefront and corner we are in
  unsigned wavefront = ((const MiniKBATask::MiniKBAArgs*)task->args)->wavefront;
  unsigned corner = ((const MiniKBATask::MiniKBAArgs*)task->args)->corner;
  assert(p[0] < int(cache_valid[corner][wavefront].size()));
  // Check to see if it is in the cache
  if (cache_valid[corner][wavefront][p[0]])
    return cache[corner][wavefront][p[0]];
  // Not in the cache, let's go find it
  Point<3> spatial_point = 
    Snap::wavefront_map[corner][wavefront][p[0]].get_point<3>();
  spatial_point += stride; 
  LogicalRegion result = runtime->get_logical_subregion_by_color(upper_bound,
                                     DomainPoint::from_point<3>(spatial_point));
  // Get the right sub-partition 
#if 0
  LogicalPartition subpartition = runtime->get_logical_partition_by_color(
                                    subregion, Snap::GHOST_X_PARTITION+dim);
  LogicalRegion result = 
    runtime->get_logical_subregion_by_color(subpartition, color);
  cache[corner][wavefront][p[0]] = result;
  cache_valid[corner][wavefront][p[0]] = true;
#endif
  return result;
}

//------------------------------------------------------------------------------
template<>
/*static*/ void AndReduction::apply<true>(LHS &lhs, RHS rhs)
//------------------------------------------------------------------------------
{
  // This is monotonic so no need for synchronization either way 
  if (!rhs)
    lhs = false;
}

//------------------------------------------------------------------------------
template<>
/*static*/ void AndReduction::apply<false>(LHS &lhs, RHS rhs)
//------------------------------------------------------------------------------
{
  // This is monotonic so no need for synchronization either way 
  if (!rhs)
    lhs = false;
}

//------------------------------------------------------------------------------
template<>
/*static*/ void AndReduction::fold<true>(RHS &rhs1, RHS rhs2)
//------------------------------------------------------------------------------
{
  // This is monotonic so no need for synchronization either way
  if (!rhs2)
    rhs1 = false;
}

//------------------------------------------------------------------------------
template<>
/*static*/ void AndReduction::fold<false>(RHS &rhs1, RHS rhs2)
//------------------------------------------------------------------------------
{
  // This is monotonic so no need for synchronization either way
  if (!rhs2)
    rhs1 = false;
}

const double SumReduction::identity = 0.0;

//------------------------------------------------------------------------------
template <>
void SumReduction::apply<true>(LHS &lhs, RHS rhs) 
//------------------------------------------------------------------------------
{
  lhs += rhs;
}

//------------------------------------------------------------------------------
template<>
void SumReduction::apply<false>(LHS &lhs, RHS rhs)
//------------------------------------------------------------------------------
{
  long *target = (long *)&lhs;
  union { long as_int; double as_float; } oldval, newval;
  do {
    oldval.as_int = *target;
    newval.as_float = oldval.as_float + rhs;
  } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
}

//------------------------------------------------------------------------------
template <>
void SumReduction::fold<true>(RHS &rhs1, RHS rhs2) 
//------------------------------------------------------------------------------
{
  rhs1 += rhs2;
}

//------------------------------------------------------------------------------
template<>
void SumReduction::fold<false>(RHS &rhs1, RHS rhs2)
//------------------------------------------------------------------------------
{
  long *target = (long *)&rhs1;
  union { long as_int; double as_float; } oldval, newval;
  do {
    oldval.as_int = *target;
    newval.as_float = oldval.as_float + rhs2;
  } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
}

const MomentQuad QuadReduction::identity = MomentQuad();

//------------------------------------------------------------------------------
template<>
void QuadReduction::apply<true>(LHS &lhs, RHS rhs)
//------------------------------------------------------------------------------
{
  for (int i = 0; i < 4; i++)
    lhs[i] += rhs[i];
}

//------------------------------------------------------------------------------
template<>
void QuadReduction::apply<false>(LHS &lhs, RHS rhs)
//------------------------------------------------------------------------------
{
  for (int i = 0; i < 4; i++)
  {
    long *target = (long *)&lhs[i];
    union { long as_int; double as_float; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_float = oldval.as_float + rhs[i];
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int)); 
  } 
}

//------------------------------------------------------------------------------
template<>
void QuadReduction::fold<true>(RHS &rhs1, RHS rhs2)
//------------------------------------------------------------------------------
{
  for (int i = 0; i < 4; i++)
    rhs1[i] += rhs2[i];
}

//------------------------------------------------------------------------------
template<>
void QuadReduction::fold<false>(RHS &rhs1, RHS rhs2)
//------------------------------------------------------------------------------
{
  for (int i = 0; i < 4; i++)
  {
    long *target = (long *)&rhs1[i];
    union { long as_int; double as_float; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_float = oldval.as_float + rhs2[i];
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int)); 
  }
}

