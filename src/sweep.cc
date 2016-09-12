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

//------------------------------------------------------------------------------
MiniKBATask::MiniKBATask(const Snap &snap, const Predicate &pred,
                         const SnapArray &flux, const SnapArray &qtot,
                         int group, int corner, bool even) 
  : SnapTask<MiniKBATask>(snap, Rect<3>(Point<3>::ZEROES(), Point<3>::ZEROES()),
                          pred), mini_kba_args(MiniKBAArgs(corner, group, even))
//------------------------------------------------------------------------------
{
  global_arg = TaskArgument(&mini_kba_args, sizeof(mini_kba_args));
  Snap::SnapFieldID group_field = SNAP_ENERGY_GROUP_FIELD(group);
  Snap::SnapProjectionID sweep_id = SNAP_SWEEP_PROJECTION(wavefront, corner);
  // If you add projection requirements here, remember to update
  // the value of NON_GHOST_REQUIREMENTS in sweep.h
  qtot.add_projection_requirement(READ_ONLY, *this, group_field, sweep_id);
  // We want to read-write our whole region for this point
  flux.add_projection_requirement(READ_WRITE, *this, group_field, sweep_id);
  // Add our reading ghost regions
  for (int i = 0; i < Snap::num_dims; i++)
  {
    Snap::SnapFieldID ghost_read = 
      SNAP_GHOST_FLUX_FIELD(corner, group, even, i);
    Snap::SnapProjectionID proj_id = SNAP_GHOST_INPUT_PROJECTION(corner, i);    
    flux.add_projection_requirement(READ_ONLY, *this, ghost_read, proj_id);
  }
  // Then add our writing ghost regions
  for (int i = 0; i < Snap::num_dims; i++)
  {
    Snap::SnapFieldID ghost_write = 
      SNAP_GHOST_FLUX_FIELD(corner, group, !even, i);
    Snap::SnapProjectionID proj_id = SNAP_GHOST_OUTPUT_PROJECTION(corner, i);
    flux.add_projection_requirement(WRITE_DISCARD, *this, ghost_write, proj_id);
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
}

//------------------------------------------------------------------------------
/*static*/ void MiniKBATask::gpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
}

