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

//------------------------------------------------------------------------------
void Snap::setup(void)
//------------------------------------------------------------------------------
{
}

//------------------------------------------------------------------------------
void Snap::transport_solve(void)
//------------------------------------------------------------------------------
{
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
  Snap snap(ctx, runtime); 
  snap.setup();
  snap.transport_solve();
  snap.output();
}

//------------------------------------------------------------------------------
/*static*/ void Snap::parse_arguments(int argc, char **argv)
//------------------------------------------------------------------------------
{
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

