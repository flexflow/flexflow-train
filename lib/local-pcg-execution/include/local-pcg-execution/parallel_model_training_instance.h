#ifndef _FLEXFLOW_LIB_LOCAL_PCG_EXECUTION_INCLUDE_LOCAL_PCG_EXECUTION_PARALLEL_MODEL_TRAINING_INSTANCE_H
#define _FLEXFLOW_LIB_LOCAL_PCG_EXECUTION_INCLUDE_LOCAL_PCG_EXECUTION_PARALLEL_MODEL_TRAINING_INSTANCE_H

#include "compiler/mapped_parallel_computation_graph.dtg.h"
#include "kernels/allocation.h"
#include "local-execution/local_atomic_tensor_backing.dtg.h"
#include "local-execution/local_task_registry.dtg.h"
#include "local-pcg-execution/local_parallel_tensor_backing.dtg.h"
#include "local-pcg-execution/task_group_execution_times.dtg.h"
#include "op-attrs/ops/loss_functions/loss_attrs.dtg.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "pcg/parallel_computation_graph/parallel_layer_guid_t.dtg.h"
#include "task-spec/runtime_task_invocation/runtime_arg_config.dtg.h"
#include "task-spec/symbolic/training_symbolic_computation_graph_from_pcg_conversion.dtg.h"

namespace FlexFlow {

struct ParallelModelTrainingInstance {
  ParallelModelTrainingInstance(Allocator const &,
                                LossAttrs const &,
                                OptimizerAttrs const &);

public:
  std::unordered_map<parallel_layer_guid_t,
                     std::optional<TaskGroupExecutionTimes>>
      forward();
  std::unordered_map<parallel_layer_guid_t,
                     std::optional<TaskGroupExecutionTimes>>
      backward();
  void update();
  GenericTensorAccessorR get_loss_tensor_accessor() const;

private:
  Allocator allocator;
  LossAttrs loss_attrs;
  OptimizerAttrs optimizer_attrs;
  TrainingSymbolicComputationGraphFromPcgConversion symbolic_cg;
  MappedParallelComputationGraph mapped_pcg;
  LocalParallelTensorBacking local_tensor_backing;
  LocalAtomicTensorBacking local_atomic_tensor_backing;
  LocalTaskRegistry local_task_registry;
  RuntimeArgConfig runtime_arg_config;
};

} // namespace FlexFlow

#endif
