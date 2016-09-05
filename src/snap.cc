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
#include "outer.h"
#include "inner.h"

#include <cstdio>

//------------------------------------------------------------------------------
void Snap::setup(void)
//------------------------------------------------------------------------------
{
#if 0
  // This is the index space for all our regions
  Rect<3> bounds(Point<3>(0,0,0), Point<3>(nx-1,ny-1,nz-1));
  IndexSpace simulation_space = 
    runtime->create_index_space(ctx, Domain::from_rect<3>(bounds));
  // Create the disjoint partition of the index space 

  // Create the ghost partitions for each subregion

  // Make a field space for all the energy groups
  FieldSpace group_space = runtime->create_field_space(ctx); 
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, group_space);
    std::vector<FieldID> group_fields(num_groups);
    for (unsigned idx = 0; idx < num_groups; idx++)
      group_fields[idx] = FID_GROUP_0 + idx;
    std::vector<size_t> group_sizes(num_groups, sizeof(double));
    allocator.allocate_fields(group_sizes, group_fields);
  }
  flux0 = runtime->create_logical_region(ctx, simulation_space, group_space);
  flux0po = runtime->create_logical_region(ctx, simulation_space, group_space);
  flux0pi = runtime->create_logical_region(ctx, simulation_space, group_space);
#endif
}

//------------------------------------------------------------------------------
void Snap::transport_solve(void)
//------------------------------------------------------------------------------
{
#if 0
  // Use a tunable variable to decide how far ahead the outer loop gets
  Future outer_runahead_future = 
    runtime->select_tunable_value(ctx, OUTER_RUNAHEAD_TUNABLE);
  Future inner_runahead_future = 
    runtime->select_tunable_value(ctx, INNER_RUNAHEAD_TUNABLE);
  const unsigned outer_runahead = outer_runahead_future.get_result<unsigned>();
  assert(outer_runahead > 0);
  const unsigned inner_runahead = inner_runahead_future.get_result<unsigned>();
  assert(inner_runahead > 0);
  // Loop over time steps
  std::deque<Future> outer_converged_tests;
  std::deque<Future> inner_converged_tests;
  // Iterate over time steps
  for (unsigned cy = 0; cy < num_steps; cy++)
  {
    outer_converged_tests.clear();
    Predicate outer_pred = Predicate::TRUE_PRED;
    // The outer solve loop    
    for (unsigned otno = 0; otno < max_outer_iters; otno++)
    {
      // Do the outer source calculation 
      CalcOuterSource outer_src(outer_pred);
      outer_src.dispatch(ctx, runtime);
      // Save the fluxes
      save_fluxes(flux0, flux0po);
      // Do the inner solve
      inner_converged_tests.clear();
      Predicate inner_pred = Predicate::TRUE_PRED;
      // The inner solve loop
      for (unsigned inno=0; inno < max_inner_iters; inno++)
      {
        // Do the inner source calculation
        CalcInnerSource inner_src(inner_pred);
        inner_src.dispatch(ctx, runtime);
        // Save the fluxes
        save_fluxes(flux0, flux0pi);
        // Perform the sweeps

        // Test for inner convergence
        TestInnerConvergence inner_conv(inner_pred);
        Future inner_converged = inner_conv.dispatch(ctx, runtime);
        inner_converged_tests.push_back(inner_converged);
        // Update the next predicate
        Predicate converged = runtime->create_predicate(ctx, inner_converged);
        inner_pred = runtime->predicate_not(ctx, converged);
        // See if we've run far enough ahead
        if (inner_converged_tests.size() == inner_runahead)
        {
          Future f = inner_converged_tests.front();
          inner_converged_tests.pop_front();
          if (f.get_result<bool>())
            break;
        }
      }
      // Test for outer convergence
      TestOuterConvergence outer_conv(outer_pred);
      Future outer_converged = outer_conv.dispatch(ctx, runtime);
      outer_converged_tests.push_back(outer_converged);
      // Update the next predicate
      Predicate converged = runtime->create_predicate(ctx, outer_converged);
      outer_pred = runtime->predicate_not(ctx, converged);
      // See if we've run far enough ahead
      if (outer_converged_tests.size() == outer_runahead)
      {
        Future f = outer_converged_tests.front();
        outer_converged_tests.pop_front();
        if (f.get_result<bool>())
          break;
      }
    }
  }
#endif
}

//------------------------------------------------------------------------------
void Snap::output(void)
//------------------------------------------------------------------------------
{
}

//------------------------------------------------------------------------------
/*static*/ void Snap::snap_top_level_task(const Task *task,
                                     const std::vector<PhysicalRegion> &regions,
                                     Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
  printf("Welcome to Legion-SNAP!\n");
  report_arguments();
#if 0
  Snap snap(ctx, runtime); 
  snap.setup();
  snap.transport_solve();
  snap.output();
#endif
}

static void skip_line(FILE *f)
{
  char buffer[80];
  fgets(buffer, 79, f);
}

static void read_int(FILE *f, const char *name, int &result)
{
  char format[80];
  sprintf(format,"  %s=%%d", name);
  fscanf(f, format, &result);
}

static void read_bool(FILE *f, const char *name, bool &result)
{
  char format[80];
  sprintf(format,"  %s=%%d", name);
  int temp = 0;
  fscanf(f, format, &temp);
  result = (temp != 0);
}

static void read_double(FILE *f, const char *name, double &result)
{
  char format[80];
  sprintf(format,"  %s=%%lf", name);
  fscanf(f, format, &result);
}

int Snap::num_dims = 3;
int Snap::nx_chunks = 1;
int Snap::ny_chunks = 1;
int Snap::nz_chunks = 1;
int Snap::nx = 4;
double Snap::lx = 0.1;
int Snap::ny = 4;
double Snap::ly = 0.1;
int Snap::nz = 4;
double Snap::lz = 0.1;
int Snap::num_moments = 1;
int Snap::num_angles = 1;
int Snap::num_groups = 1;
double Snap::convergence_eps = 1e-2;
int Snap::max_inner_iters = 1;
int Snap::max_outer_iters = 1;
bool Snap::time_dependent = false;
double Snap::total_sim_time = 0.1;
int Snap::num_steps = 2;
Snap::MaterialLayout Snap::material_layout = HOMOGENEOUS_LAYOUT;
Snap::SourceLayout Snap::source_layout = EVERYWHERE_SOURCE; 
bool Snap::dump_scatter = false;
bool Snap::dump_iteration = false;
int Snap::dump_flux = 0;
bool Snap::flux_fixup = false;
bool Snap::dump_solution = false;
int Snap::dump_kplane = 0;
int Snap::dump_population = 0;
bool Snap::minikba_sweep = false;
bool Snap::single_angle_copy = false;

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
  read_bool(f, "solutp", dump_solution);
  read_int(f, "kplane", dump_kplane);
  read_int(f, "popout", dump_population);
  read_bool(f, "swp_typ", minikba_sweep);
  read_bool(f, "angcpy", single_angle_copy);
  fclose(f);
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
  assert((1 <= num_moments) && (num_moments <= 4));
  assert(1 <= num_angles);
  assert(1 <= num_groups);
  assert((0.0 < convergence_eps) && (convergence_eps < 1e-2));
  assert(1 <= max_inner_iters);
  assert(1 <= max_outer_iters);
  if (time_dependent)
    assert(0.0 <= total_sim_time);
  assert(1 <= num_steps);
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
/*static*/ void Snap::register_task_variants(void)
//------------------------------------------------------------------------------
{
  TaskVariantRegistrar registrar(SNAP_TOP_LEVEL_TASK_ID, "snap_main_variant");
  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  Runtime::preregister_task_variant<snap_top_level_task>(registrar,"snap_main");
  Runtime::set_top_level_task_id(SNAP_TOP_LEVEL_TASK_ID);
  Runtime::set_registration_callback(mapper_registration);
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

