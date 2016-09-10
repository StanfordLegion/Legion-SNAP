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
#include <cstring>

#ifndef SNAP_MAX_ENERGY_GROUPS
#define SNAP_MAX_ENERGY_GROUPS            8192
#endif

using namespace Legion;
using namespace Legion::Mapping;
using namespace LegionRuntime::Arrays;

extern LegionRuntime::Logger::Category log_snap;

class Snap {
public:
  enum SnapTaskID {
    SNAP_TOP_LEVEL_TASK_ID,
    CALC_OUTER_SOURCE_TASK_ID,
    TEST_OUTER_CONVERGENCE_TASK_ID,
    CALC_INNER_SOURCE_TASK_ID,
    TEST_INNER_CONVERGENCE_TASK_ID,
    LAST_TASK_ID, // must be last
  };
#define SNAP_TASK_NAMES           \
    "Top Level Task",             \
    "Calc Outer Source",          \
    "Test Outer Convergence",     \
    "Calc Inner Source",          \
    "Test Inner Convergence",
  static const char* task_names[LAST_TASK_ID];
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
  enum SnapReductionID {
    NO_REDUCTION_ID = 0,
    AND_REDUCTION_ID = 1,
  };
  enum SnapFieldID {
    FID_GROUP_0,
    FID_GROUP_MAX = FID_GROUP_0 + SNAP_MAX_ENERGY_GROUPS,
  };
  enum SnapPartitionID {
    DISJOINT_PARTITION,
    GHOST_X_PARTITION,
    GHOST_Y_PARTITION,
    GHOST_Z_PARTITION,
  };
public:
  Snap(Context c, Runtime *rt)
    : ctx(c), runtime(rt) { }
public:
  inline const Rect<3>& get_simulation_bounds(void) const 
    { return simulation_bounds; }
  inline const Rect<3>& get_launch_bounds(void) const
    { return launch_bounds; }
public:
  void setup(void);
  void transport_solve(void);
  void output(void);
private:
  const Context ctx;
  Runtime *const runtime;
private:
  // Simulation bounds
  Rect<3> simulation_bounds;
  Rect<3> launch_bounds;
  // Primary logical regions
  LogicalRegion flux0, flux0po, flux0pi;  
public:
  static void snap_top_level_task(const Task *task,
                                  const std::vector<PhysicalRegion> &regions,
                                  Context ctx, Runtime *runtime);
  static void save_fluxes(LogicalRegion source, LogicalRegion target);
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

template<typename T>
class SnapTask : public IndexLauncher {
public:
  SnapTask(const Snap &snap, const Rect<3> &launch_domain, const Predicate &pred)
    : IndexLauncher(T::TASK_ID, Domain::from_rect<3>(launch_domain), 
                    TaskArgument(), ArgumentMap(), pred) { }
public:
  void dispatch(Context ctx, Runtime *runtime)
  { 
    log_snap.info("Dispatching Task %s (ID %d)", 
        Snap::task_names[T::TASK_ID], T::TASK_ID);
    runtime->execute_index_space(ctx, *this);
  }
  template<typename REDOP>
  Future dispatch(Context ctx, Runtime *runtime)
  {
    assert(REDOP::REDOP_ID == T::REDOP);
    log_snap.info("Dispatching Task %s (ID %d) with Reduction %d", 
                  Snap::task_names[T::TASK_ID], T::TASK_ID, T::REDOP);
    return runtime->execute_index_space(ctx, *this, T::REDOP);
  }
public:
  static void preregister_all_variants(void)
  {
    T::preregister_cpu_variants();
    T::preregister_gpu_variants();
  }
  static void register_task_name(Runtime *runtime)
  {
    runtime->attach_name(T::TASK_ID, Snap::task_names[T::TASK_ID]);
  }
private:
  template<void (*TASK_PTR)(const Task*,
      const std::vector<PhysicalRegion>&, Context, Runtime*)>
  static void snap_task_wrapper(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
  {
    log_snap.info("Running Task %s (UID %lld) on Processor " IDFMT "",
        task->get_task_name(), task->get_unique_id(), 
        runtime->get_executing_processor(ctx).id);
    (*TASK_PTR)(task, regions, ctx, runtime);
  }
  template<typename RET_T, RET_T (*TASK_PTR)(const Task*,
      const std::vector<PhysicalRegion>&, Context, Runtime*)>
  static RET_T snap_task_wrapper(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
  {
    log_snap.info("Running Task %s (UID %lld) on Processor " IDFMT "",
        task->get_task_name(), task->get_unique_id(), 
        runtime->get_executing_processor(ctx).id);
    RET_T result = (*TASK_PTR)(task, regions, ctx, runtime);
    return result;
  }
protected:
  // For registering CPU variants
  template<void (*TASK_PTR)(const Task*,
      const std::vector<PhysicalRegion>&, Context, Runtime*)>
  static void register_cpu_variant(void)
  {
    char variant_name[128];
    strcpy(variant_name, "CPU ");
    strncat(variant_name, Snap::task_names[T::TASK_ID], 123);
    TaskVariantRegistrar registrar(T::TASK_ID, true/*global*/,
        NULL/*generator*/, variant_name);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<snap_task_wrapper<TASK_PTR> >(registrar, 
                                                Snap::task_names[T::TASK_ID]);
  }
  template<typename RET_T, RET_T (*TASK_PTR)(const Task*,
      const std::vector<PhysicalRegion>&, Context, Runtime*)>
  static void register_cpu_variant(void)
  {
    char variant_name[128];
    strcpy(variant_name, "CPU ");
    strncat(variant_name, Snap::task_names[T::TASK_ID], 123);
    TaskVariantRegistrar registrar(T::TASK_ID, true/*global*/,
        NULL/*generator*/, variant_name);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<RET_T,snap_task_wrapper<RET_T,TASK_PTR> >(
                                       registrar, Snap::task_names[T::TASK_ID]);
  }
protected:
  // For registering GPU variants
  template<void (*TASK_PTR)(const Task*,
      const std::vector<PhysicalRegion>&, Context, Runtime*)>
  static void register_gpu_variant(void)
  {
    char variant_name[128];
    strcpy(variant_name, "CPU ");
    strncat(variant_name, Snap::task_names[T::TASK_ID], 123);
    TaskVariantRegistrar registrar(T::TASK_ID, true/*global*/,
        NULL/*generator*/, variant_name);
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    Runtime::preregister_task_variant<snap_task_wrapper<TASK_PTR> >(registrar,
                                                Snap::task_names[T::TASK_ID]);
  }
  template<typename RET_T, RET_T (*TASK_PTR)(const Task*,
      const std::vector<PhysicalRegion>&, Context, Runtime*)>
  static void register_gpu_variant(void)
  {
    char variant_name[128];
    strcpy(variant_name, "CPU ");
    strncat(variant_name, Snap::task_names[T::TASK_ID], 123);
    TaskVariantRegistrar registrar(T::TASK_ID, true/*global*/,
        NULL/*generator*/, variant_name);
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    Runtime::preregister_task_variant<RET_T,snap_task_wrapper<RET_T,TASK_PTR> >(
                                       registrar, Snap::task_names[T::TASK_ID]);
  }
};

class AndReduction {
public:
  static const Snap::SnapReductionID REDOP_ID = Snap::AND_REDUCTION_ID;
public:
  typedef bool LHS;
  typedef bool RHS;
  static const bool identity = true;
public:
  template<bool EXCLUSIVE> static void apply(LHS &lhs, RHS rhs);
  template<bool EXCLUSIVE> static void fold(RHS &rhs1, RHS rhs2);
};

#endif // __SNAP_H__

