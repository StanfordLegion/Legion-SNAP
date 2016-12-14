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
        // Should be enough to always unroll the inner loop
        runtime->pack_tunable<unsigned>(8, output);
        break;
      }
    case SWEEP_ENERGY_CHUNKS_TUNABLE:
      {
        // 8 directions * number of energy fields should be larger
        // then the number of processors in a node since we use field 
        // parallelism to keep all the processors in a node busy 
        if (local_gpus.empty()) {
          // Mapping to CPUs only
          const int num_cpus = local_cpus.size();
          int result = 8/*directions*/ * Snap::num_groups / num_cpus;
          // Clamp it at the number of groups if necessary
          if (result > Snap::num_groups)
            result = Snap::num_groups;
          runtime->pack_tunable<int>(result, output);
        } else {
          // Mapping to GPUs
          const int num_gpus = local_gpus.size();
          int result = 8/*directions*/ * Snap::num_groups / num_gpus;
          // Clamp it at the number of groups if necessary
          if (result > Snap::num_groups)
            result = Snap::num_groups;
          runtime->pack_tunable<int>(result, output);
        }
        break;
      }
    default:
      // Fall back to the default mapper
      DefaultMapper::select_tunable_value(ctx, task, input, output);
  }
}

//------------------------------------------------------------------------------
void Snap::SnapMapper::map_task(const MapperContext ctx,
                                const Task &task,
                                const MapTaskInput &input,
                                      MapTaskOutput &output)
//------------------------------------------------------------------------------
{
  switch (task.task_id)
  {
    case CALC_OUTER_SOURCE_TASK_ID:
      {
          
      }
    case TEST_OUTER_CONVERGENCE_TASK_ID:
      {

      }
    case CALC_INNER_SOURCE_TASK_ID:
      {

      }
    case TEST_INNER_CONVERGENCE_TASK_ID:
      {

      }
    case MINI_KBA_TASK_ID:
      {

      }
    case EXPAND_CROSS_SECTION_TASK_ID:
      {

      }
    case EXPAND_SCATTERING_CROSS_SECTION_TASK_ID:
      {

      }
    case CALCULATE_GEOMETRY_PARAM_TASK_ID:
      {

      }
    case MMS_SCALE_TASK_ID:
      {

      }
    case BIND_INNER_CONVERGENCE_TASK_ID:
    case BIND_OUTER_CONVERGENCE_TASK_ID:
      {

      }
    default:
      DefaultMapper::map_task(ctx, task, input, output);
  }
}

