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

#ifndef __SNAP_H__
#define __SNAP_H__

#include "legion.h"
#include "default_mapper.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>

using namespace Legion;
using namespace Legion::Mapping;

class Snap {
public:
  enum SnapTaskIDs {
    SNAP_TOP_LEVEL_TASK_ID,
  };
  enum MaterialLayout {
    HOMOGENEOUS_LAYOUT = 0,
    CENTER_LAYOUT = 1,
    CORNER_LAYOUT = 2,
  };
  enum SourceLayout {
    EVERYWHERE_SOURCE = 0,
    CENTER_SOURCE = 1,
    CORNER_SOURCE = 2,
    MMS_SOURCE = 3,
  };
  enum SnapTunable {
    OUTER_RUNAHEAD_TUNABLE = DefaultMapper::DEFAULT_TUNABLE_LAST,
    INNER_RUNAHEAD_TUNABLE = DefaultMapper::DEFAULT_TUNABLE_LAST+1,
  };
public:
  Snap(Context c, Runtime *rt)
    : ctx(c), runtime(rt) { }
public:
  void setup(void);
  void transport_solve(void);
  void output(void);
private:
  const Context ctx;
  Runtime *const runtime;
private:
  // Primary logical regions
  // Assuming only one material right now
  LogicalRegion velocity; // 1-D number of groups
  LogicalRegion materials; // nx-ny-nz material identifier array
  LogicalRegion qi; // nx-ny-nz-ng fixed source array for src_opt < 3
  LogicalRegion qim; // nang-nx-ny-nz-noct,ng fixed source array for src_opt >= 3
  LogicalRegion sigt; // 1-D number of groups total interaction
  LogicalRegion siga; // 1-D number of gropus absorbtion
  LogicalRegion sigs; // 1-D number of groups scatering
  LogicalRegion slgg; // nmom-ng-ng scatering matrix
  LogicalRegion vdelt; // 1-D number of groups time-absorbtion coefficient
public:
  static void snap_top_level_task(const Task *task,
                                  const std::vector<PhysicalRegion> &regions,
                                  Context ctx, Runtime *runtime);
public:
  static void parse_arguments(int argc, char **argv);
  static void report_arguments(void);
  static void register_task_variants(void);
  static void mapper_registration(Machine machine, Runtime *runtime,
                                  const std::set<Processor> &local_procs);
public:
  // Configuration parameters read from input file
  static int num_dims; // originally ndimen 1-3
  static int nx_chunks; // originally ichunk 1 <= # <= nx
  static int ny_chunks; // originally npey 1 <= # <= ny
  static int nz_chunks; // originally npez 1 <= # <= nz
  static int nx; // 4 <= #
  static double lx; // 0.0 < #
  static int ny; // 4 <= #
  static double ly; // 0.0 < #
  static int nz; // 4 <= #
  static double lz; // 0.0 < #
  static int num_moments; // originally nmom 1 <= # <= 4
  static int num_angles; // originally nang 1 <= #
  static int num_groups; // originally ng 1 <= #
  static double convergence_eps; // originally epsi 0.0 < # < 1e-2
  static int max_inner_iters; // originally iitm 1 <= # 
  static int max_outer_iters; // originally oitm 1 <= #
  static bool time_dependent; // originally timedep
  static double total_sim_time; // originally tf 0.0 <= # if time dependent
  static int num_steps; // originally nsteps 1 <= #
  static MaterialLayout material_layout; // originally mat_opt
  static SourceLayout source_layout; // originally src_opt
  static bool dump_scatter; // originally scatp
  static bool dump_iteration; // originally it_dep
  static int dump_flux; // originally fluxp 0,1,2
  static bool flux_fixup; // originally fixup
  static bool dump_solution; // originally soloutp
  static int dump_kplane; // originally kplane 0,1,2
  static int dump_population;  // originally popout
  static bool minikba_sweep; // originally swp_typ
  static bool single_angle_copy; // originall angcpy
public:
  // Snap mapper derived from the default mapper
  class SnapMapper : public Legion::Mapping::DefaultMapper {
  public:
    SnapMapper(MapperRuntime *rt, Machine machine, Processor local,
               const char *mapper_name);
  public:
    virtual void select_tunable_value(const MapperContext ctx,
                                      const Task& task,
                                      const SelectTunableInput& input,
                                            SelectTunableOutput& output);
  };
};

#endif // __SNAP_H__

