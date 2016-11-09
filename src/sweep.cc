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

#endif
}

//------------------------------------------------------------------------------
/*static*/ void MiniKBATask::gpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE

#endif
}

