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
#define SNAP_MAX_ENERGY_GROUPS            1024
#endif
#define MINI_KBA_NON_GHOST_REQUIREMENTS   10

#ifndef PI
#define PI (3.14159265358979)
#endif

using namespace Legion;
using namespace Legion::Mapping;
using namespace LegionRuntime::Arrays;

extern LegionRuntime::Logger::Category log_snap;

class SnapArray;

class Snap {
public:
  enum SnapTaskID {
    SNAP_TOP_LEVEL_TASK_ID,
    INIT_MATERIAL_TASK_ID,
    INIT_SOURCE_TASK_ID,
    CALC_OUTER_SOURCE_TASK_ID,
    TEST_OUTER_CONVERGENCE_TASK_ID,
    CALC_INNER_SOURCE_TASK_ID,
    TEST_INNER_CONVERGENCE_TASK_ID,
    MINI_KBA_TASK_ID,
    EXPAND_CROSS_SECTION_TASK_ID,
    EXPAND_SCATTERING_CROSS_SECTION_TASK_ID,
    CALCULATE_GEOMETRY_PARAM_TASK_ID,
    MMS_INIT_FLUX_TASK_ID,
    MMS_INIT_SOURCE_TASK_ID,
    MMS_INIT_TIME_DEPENDENT_TASK_ID,
    MMS_SCALE_TASK_ID,
    MMS_COMPARE_TASK_ID,
    LAST_TASK_ID, // must be last
  };
#define SNAP_TASK_NAMES                 \
    "Top Level Task",                   \
    "Initialize Material",              \
    "Initialize Source",                \
    "Calc Outer Source",                \
    "Test Outer Convergence",           \
    "Calc Inner Source",                \
    "Test Inner Convergence",           \
    "Mini KBA",                         \
    "Expand Cross Section",             \
    "Expand Scattering Cross Section",  \
    "Calcuate Geometry Param",          \
    "MMS Init Flux",                    \
    "MMS Init Source",                  \
    "MMS Init Time Dependent",          \
    "MMS Scale",                        \
    "MMS Compare"
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
    SUM_REDUCTION_ID = 2,
    QUAD_REDUCTION_ID = 3,
    MMS_REDUCTION_ID = 4,
  };
  enum SnapFieldID {
    FID_SINGLE = 0, // For field spaces with just one field
    // Fields for energy groups
    FID_GROUP_0 = FID_SINGLE,
    // ...
    FID_GROUP_MAX = FID_GROUP_0 + SNAP_MAX_ENERGY_GROUPS,
    FID_CORNER_0_GHOST_FLUX_X_0_EVEN = FID_GROUP_MAX,
    FID_CORNER_0_GHOST_FLUX_Y_0_EVEN,
    FID_CORNER_0_GHOST_FLUX_Z_0_EVEN,
    FID_CORNER_1_GHOST_FLUX_X_0_EVEN,
    FID_CORNER_1_GHOST_FLUX_Y_0_EVEN,
    FID_CORNER_1_GHOST_FLUX_Z_0_EVEN,
    FID_CORNER_2_GHOST_FLUX_X_0_EVEN,
    FID_CORNER_2_GHOST_FLUX_Y_0_EVEN,
    FID_CORNER_2_GHOST_FLUX_Z_0_EVEN,
    FID_CORNER_3_GHOST_FLUX_X_0_EVEN,
    FID_CORNER_3_GHOST_FLUX_Y_0_EVEN,
    FID_CORNER_3_GHOST_FLUX_Z_0_EVEN,
    FID_CORNER_4_GHOST_FLUX_X_0_EVEN,
    FID_CORNER_4_GHOST_FLUX_Y_0_EVEN,
    FID_CORNER_4_GHOST_FLUX_Z_0_EVEN,
    FID_CORNER_5_GHOST_FLUX_X_0_EVEN,
    FID_CORNER_5_GHOST_FLUX_Y_0_EVEN,
    FID_CORNER_5_GHOST_FLUX_Z_0_EVEN,
    FID_CORNER_6_GHOST_FLUX_X_0_EVEN,
    FID_CORNER_6_GHOST_FLUX_Y_0_EVEN,
    FID_CORNER_6_GHOST_FLUX_Z_0_EVEN,
    FID_CORNER_7_GHOST_FLUX_X_0_EVEN,
    FID_CORNER_7_GHOST_FLUX_Y_0_EVEN,
    FID_CORNER_7_GHOST_FLUX_Z_0_EVEN,

    FID_CORNER_0_GHOST_FLUX_X_0_ODD,
    FID_CORNER_0_GHOST_FLUX_Y_0_ODD,
    FID_CORNER_0_GHOST_FLUX_Z_0_ODD,
    FID_CORNER_1_GHOST_FLUX_X_0_ODD,
    FID_CORNER_1_GHOST_FLUX_Y_0_ODD,
    FID_CORNER_1_GHOST_FLUX_Z_0_ODD,
    FID_CORNER_2_GHOST_FLUX_X_0_ODD,
    FID_CORNER_2_GHOST_FLUX_Y_0_ODD,
    FID_CORNER_2_GHOST_FLUX_Z_0_ODD,
    FID_CORNER_3_GHOST_FLUX_X_0_ODD,
    FID_CORNER_3_GHOST_FLUX_Y_0_ODD,
    FID_CORNER_3_GHOST_FLUX_Z_0_ODD,
    FID_CORNER_4_GHOST_FLUX_X_0_ODD,
    FID_CORNER_4_GHOST_FLUX_Y_0_ODD,
    FID_CORNER_4_GHOST_FLUX_Z_0_ODD,
    FID_CORNER_5_GHOST_FLUX_X_0_ODD,
    FID_CORNER_5_GHOST_FLUX_Y_0_ODD,
    FID_CORNER_5_GHOST_FLUX_Z_0_ODD,
    FID_CORNER_6_GHOST_FLUX_X_0_ODD,
    FID_CORNER_6_GHOST_FLUX_Y_0_ODD,
    FID_CORNER_6_GHOST_FLUX_Z_0_ODD,
    FID_CORNER_7_GHOST_FLUX_X_0_ODD,
    FID_CORNER_7_GHOST_FLUX_Y_0_ODD,
    FID_CORNER_7_GHOST_FLUX_Z_0_ODD,

    FID_MAX_CORNER_GHOST_FLUX_FIELDS = 
      FID_CORNER_0_GHOST_FLUX_X_0_EVEN + 48 * SNAP_MAX_ENERGY_GROUPS,
  };
#define SNAP_ENERGY_GROUP_FIELD(group)    \
  ((Snap::SnapFieldID)(Snap::FID_GROUP_0 + (group)))
#define SNAP_GHOST_FLUX_FIELD_EVEN(group, corner, dim)                          \
  ((Snap::SnapFieldID)(Snap::FID_CORNER_0_GHOST_FLUX_X_0_EVEN +                 \
    ((group) * 48) + ((corner) * 3) + (dim)))
#define SNAP_GHOST_FLUX_FIELD_ODD(group, corner, dim)                           \
  ((Snap::SnapFieldID)(Snap::FID_CORNER_0_GHOST_FLUX_X_0_ODD +                  \
    ((group) * 48) + ((corner) * 3) + (dim)))
  enum SnapPartitionID {
    DISJOINT_PARTITION = 0,
    GHOST_X_PARTITION = 1,
    GHOST_Y_PARTITION = 2,
    GHOST_Z_PARTITION = 3,
  };
  enum SnapProjectionID {
    SWEEP_PROJECTION = 1,
    MINUS_X_PROJECTION = 2,
    PLUS_X_PROJECTION = 3,
    MINUX_Y_PROJECTION = 4,
    PLUS_Y_PROJECTION = 5,
    MINUS_Z_PROJECTION = 6,
    PLUS_Z_PROJECTION = 7,
  };
#define SNAP_SWEEP_PROJECTION(corner)                                     \
  ((Snap::SnapProjectionID)(Snap::WAVEFRONT_0_CORNER_0_SWEEP_PROJECTION + \
    ((wavefront) * 14) + (corner)))
#define SNAP_GHOST_PROJECTION(dim, offset)                                \
  ((Snap::SnapProjectionID)(Snap::PLUS_X_PROJECTION + ((dim) * 2) + (offset)))
  enum SnapGhostColor {
    LO_GHOST = 0,
    HI_GHOST = 1,
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
protected:
  void initialize_scattering(const SnapArray &sigt, const SnapArray &siga,
                             const SnapArray &sigs, const SnapArray &slgg) const;
  void initialize_velocity(const SnapArray &vel, const SnapArray &vdelt) const;
  void save_fluxes(const Predicate &pred,
                   const SnapArray &src, const SnapArray &dst) const;
  void perform_sweeps(const Predicate &pred, const SnapArray &flux,
                      const SnapArray &fluxm, const SnapArray &qtot, 
                      const SnapArray &vdelt, const SnapArray &dinv, 
                      const SnapArray &t_xs, SnapArray *time_flux_in[8], 
                      SnapArray *time_flux_out[8]) const;
private:
  const Context ctx;
  Runtime *const runtime;
private:
  // Simulation bounds
  Rect<3> simulation_bounds;
  Rect<3> launch_bounds;
private:
  IndexSpace simulation_is;
  IndexPartition spatial_ip;
  IndexSpace material_is;
  IndexSpace slgg_is;
  IndexSpace point_is;
private:
  FieldSpace group_fs;
  FieldSpace group_and_ghost_fs;
  FieldSpace moment_fs;
  FieldSpace flux_moment_fs;
  FieldSpace mat_fs;
  FieldSpace angle_fs;
private:
  std::vector<Domain> wavefront_domains[8];
public:
  static void snap_top_level_task(const Task *task,
                                  const std::vector<PhysicalRegion> &regions,
                                  Context ctx, Runtime *runtime); 
public:
  static void parse_arguments(int argc, char **argv);
  static void compute_wavefronts(void);
  static void compute_derived_globals(void);
  static void report_arguments(void);
  static void perform_registrations(void);
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
  static bool single_angle_copy; // originally angcpy
public: // derived
  static int num_corners; // orignally ncor
  static int nx_per_chunk;
  static int ny_per_chunk;
  static int nz_per_chunk;
  // Indexed by wavefront number and the point number
  static std::vector<std::vector<DomainPoint> > wavefront_map[8];
  // Assume all chunks are the same size, original SNAP assumes this too
  static std::vector<std::vector<Point<3> > > chunk_wavefronts;
public:
  static double dt; 
  static int cmom;
  static int num_octants;
  static double hi, hj, hk;
  static double *mu; // num angles
  static double *w; // num angles
  static double *wmu; // num angles
  static double *eta; // num angles
  static double *weta; // num angles
  static double *xi; // num angles
  static double *wxi; // num angles
  static double *ec; // num angles x num moments x num_octants
  static double *dinv; // num_angles x nx x ny x nz x 
  static int lma[4];
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

template<typename T, Snap::SnapTaskID TASK_ID> 
class SnapTask : public IndexLauncher {
public:
  SnapTask(const Snap &snap, const Rect<3> &launch_domain, const Predicate &pred)
    : IndexLauncher(TASK_ID, Domain::from_rect<3>(launch_domain), 
                    TaskArgument(), ArgumentMap(), pred) { }
public:
  void dispatch(Context ctx, Runtime *runtime, bool block = false)
  { 
    log_snap.info("Dispatching Task %s (ID %d)", 
        Snap::task_names[TASK_ID], TASK_ID);
    if (block) {
      FutureMap fm = runtime->execute_index_space(ctx, *this);
      fm.wait_all_results(true/*silence warnings*/);
    } else
      runtime->execute_index_space(ctx, *this);
  }
  template<typename OP>
  Future dispatch(Context ctx, Runtime *runtime, bool block = false)
  {
    log_snap.info("Dispatching Task %s (ID %d) with Reduction %d", 
                  Snap::task_names[TASK_ID], TASK_ID, OP::REDOP);
    if (block) {
      Future f = runtime->execute_index_space(ctx, *this, OP::REDOP);
      f.get_void_result(true/*silence warnings*/);
      return f;
    } else
      return runtime->execute_index_space(ctx, *this, OP::REDOP);
  }
public:
  static void preregister_all_variants(void)
  {
    T::preregister_cpu_variants();
    T::preregister_gpu_variants();
  }
  static void register_task_name(Runtime *runtime)
  {
    runtime->attach_name(TASK_ID, Snap::task_names[TASK_ID]);
  }
public:
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
  static void register_cpu_variant(bool leaf = false, bool inner = false)
  {
    char variant_name[128];
    strcpy(variant_name, "CPU ");
    strncat(variant_name, Snap::task_names[TASK_ID], 123);
    TaskVariantRegistrar registrar(TASK_ID, true/*global*/,
        NULL/*generator*/, variant_name);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.leaf_variant = leaf;
    registrar.inner_variant = inner;
    Runtime::preregister_task_variant<
      SnapTask<T,TASK_ID>::template snap_task_wrapper<TASK_PTR> >(
          registrar, Snap::task_names[TASK_ID]);
  }
  template<typename RET_T, RET_T (*TASK_PTR)(const Task*,
      const std::vector<PhysicalRegion>&, Context, Runtime*)>
  static void register_cpu_variant(bool leaf = false, bool inner = false)
  {
    char variant_name[128];
    strcpy(variant_name, "CPU ");
    strncat(variant_name, Snap::task_names[TASK_ID], 123);
    TaskVariantRegistrar registrar(TASK_ID, true/*global*/,
        NULL/*generator*/, variant_name);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.leaf_variant = leaf;
    registrar.inner_variant = inner;
    Runtime::preregister_task_variant<RET_T,
      SnapTask<T,TASK_ID>::template snap_task_wrapper<RET_T,TASK_PTR> >(
                                         registrar, Snap::task_names[TASK_ID]);
  }
protected:
  // For registering GPU variants
  template<void (*TASK_PTR)(const Task*,
      const std::vector<PhysicalRegion>&, Context, Runtime*)>
  static void register_gpu_variant(bool leaf = false, bool inner = false)
  {
    char variant_name[128];
    strcpy(variant_name, "GPU ");
    strncat(variant_name, Snap::task_names[TASK_ID], 123);
    TaskVariantRegistrar registrar(TASK_ID, true/*global*/,
        NULL/*generator*/, variant_name);
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.leaf_variant = leaf;
    registrar.inner_variant = inner;
    Runtime::preregister_task_variant<
      SnapTask<T,TASK_ID>::template snap_task_wrapper<TASK_PTR> >(
          registrar, Snap::task_names[TASK_ID]);
  }
  template<typename RET_T, RET_T (*TASK_PTR)(const Task*,
      const std::vector<PhysicalRegion>&, Context, Runtime*)>
  static void register_gpu_variant(bool leaf = false, bool inner = false)
  {
    char variant_name[128];
    strcpy(variant_name, "GPU ");
    strncat(variant_name, Snap::task_names[TASK_ID], 123);
    TaskVariantRegistrar registrar(TASK_ID, true/*global*/,
        NULL/*generator*/, variant_name);
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.leaf_variant = leaf;
    registrar.inner_variant = inner;
    Runtime::preregister_task_variant<RET_T,
      SnapTask<T,TASK_ID>::template snap_task_wrapper<RET_T,TASK_PTR> >(
                                         registrar, Snap::task_names[TASK_ID]);
  }
};

class SnapArray {
public:
  SnapArray(IndexSpace is, IndexPartition ip, FieldSpace fs, 
            Context ctx, Runtime *runtime, const char *name);
  ~SnapArray(void);
private:
  SnapArray(const SnapArray &rhs);
  SnapArray& operator=(const SnapArray &rhs);
public:
  inline LogicalRegion get_region(void) const { return lr; }
  inline LogicalPartition get_partition(void) const { return lp; }
  inline const std::set<FieldID>& get_regular_fields(void) const 
    { return regular_fields; }
  LogicalRegion get_subregion(const DomainPoint &color) const;
public:
  void initialize(void) const;
  template<typename T>
  void initialize(T value) const;
  PhysicalRegion map(void) const;
  void unmap(const PhysicalRegion &region) const;
public:
  template<typename T>
  inline void add_projection_requirement(PrivilegeMode priv,
                                         T &launcher) const
  {
    launcher.add_region_requirement(RegionRequirement(lp, 0/*proj id*/,
                                                      priv, EXCLUSIVE, lr));
    launcher.region_requirements.back().privilege_fields = regular_fields;
  }
  template<typename T>
  inline void add_projection_requirement(PrivilegeMode priv, T &launcher,
                        Snap::SnapFieldID field, ProjectionID proj_id = 0) const
  {
    launcher.add_region_requirement(RegionRequirement(lp, proj_id, priv,
                                                      EXCLUSIVE, lr));
    launcher.region_requirements.back().privilege_fields.insert(field);
  }
  template<typename T>
  inline void add_projection_requirement(Snap::SnapReductionID reduction,
      T &launcher, Snap::SnapFieldID field, ProjectionID proj_id = 0) const
  {
    launcher.add_region_requirement(RegionRequirement(lp, proj_id, reduction,
                                                      EXCLUSIVE, lr));
    launcher.region_requirements.back().privilege_fields.insert(field);
  }
  template<typename T>
  inline void add_region_requirement(PrivilegeMode priv,
                                     T &launcher) const
  {
    launcher.add_region_requirement(RegionRequirement(lr, priv, EXCLUSIVE, lr));
    launcher.region_requirements.back().privilege_fields = regular_fields;
  }
  template<typename T>
  inline void add_region_requirement(PrivilegeMode priv, T&launcher,
                                     Snap::SnapFieldID field) const
  {
    launcher.add_region_requirement(RegionRequirement(lr, priv, EXCLUSIVE, lr));
    launcher.region_requirements.back().privilege_fields.insert(field);
  }
protected:
  const Context ctx;
  Runtime *const runtime;
protected:
  LogicalRegion lr;
  LogicalPartition lp;
  std::set<FieldID> regular_fields;
  std::set<FieldID> ghost_fields;
  mutable std::map<DomainPoint,LogicalRegion> subregions;
};

class SnapSweepProjectionFunctor : public ProjectionFunctor {
public:
  SnapSweepProjectionFunctor(void);
public:
  virtual LogicalRegion project(Context ctx, Task *task,
                                unsigned index, 
                                LogicalRegion upper_bound,
                                const DomainPoint &point);
  virtual LogicalRegion project(Context ctx, Task *task,
                                unsigned index, 
                                LogicalPartition upper_bound,
                                const DomainPoint &point);
  virtual unsigned get_depth(void) const { return 0; }
protected:
  // Indexed by corner, wavefront, region requirement, point
  std::vector<std::vector<LogicalRegion> > 
    cache[8][MINI_KBA_NON_GHOST_REQUIREMENTS];    
  std::vector<std::vector<bool> > 
    cache_valid[8][MINI_KBA_NON_GHOST_REQUIREMENTS];
};

class SnapGhostProjectionFunctor : public ProjectionFunctor {
public:
  SnapGhostProjectionFunctor(int dim, int offset);
public:
  virtual LogicalRegion project(Context ctx, Task *task,
                                unsigned index, 
                                LogicalRegion upper_bound,
                                const DomainPoint &point);
  virtual LogicalRegion project(Context ctx, Task *task,
                                unsigned index, 
                                LogicalPartition upper_bound,
                                const DomainPoint &point);
  virtual unsigned get_depth(void) const { return 0; }
private:
  static Point<3> get_stride(int dim, int offset);
protected:
  const int dim;
  const int offset;
  const Snap::SnapGhostColor color;
  const Point<3> stride;
  // Indexed by corner, wavefront, point
  // No need for region requirement because we know the
  // tree will always be the same
  std::vector<std::vector<LogicalRegion> > cache[8];   
  std::vector<std::vector<bool> > cache_valid[8];
};

class AndReduction {
public:
  static const Snap::SnapReductionID REDOP = Snap::AND_REDUCTION_ID;
public:
  typedef bool LHS;
  typedef bool RHS;
  static const bool identity = true;
public:
  template<bool EXCLUSIVE> static void apply(LHS &lhs, RHS rhs);
  template<bool EXCLUSIVE> static void fold(RHS &rhs1, RHS rhs2);
};

class SumReduction {
public:
  static const Snap::SnapReductionID REDOP = Snap::SUM_REDUCTION_ID;
public:
  typedef double LHS;
  typedef double RHS;
  static const double identity;
public:
  template<bool EXCLUSIVE> static void apply(LHS &lhs, RHS rhs);
  template<bool EXCLUSIVE> static void fold(RHS &rhs1, RHS rhs2);
};

struct MomentTriple {
public:
  MomentTriple(double x = 0.0, double y = 0.0, double z = 0.0)
    { vals[0] = x; vals[1] = y; vals[2] = z; }
public:
  double& operator[](const int index) { return vals[index]; }
  const double& operator[](const int index) const { return vals[index]; }
public:
  double vals[3];
};

struct MomentQuad {
public:
  MomentQuad(double x = 0.0, double y = 0.0, double z = 0.0, double w = 0.0)
    { vals[0] = x; vals[1] = y; vals[2] = z; vals[3] = w; }
public:
  double& operator[](const int index) { return vals[index]; }
  const double& operator[](const int index) const { return vals[index]; }
public:
  double vals[4];
};

class QuadReduction {
public:
  static const Snap::SnapReductionID REDOP = Snap::QUAD_REDUCTION_ID;
public:
  typedef MomentQuad LHS;
  typedef MomentQuad RHS;
  static const MomentQuad identity;
public:
  template<bool EXCLUSIVE> static void apply(LHS &lhs, RHS rhs);
  template<bool EXCLUSIVE> static void fold(RHS &rhs1, RHS rhs2);
};

#endif // __SNAP_H__

