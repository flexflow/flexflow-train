#ifndef _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_EXECUTE_TASK_FOR_LAYER_H
#define _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_EXECUTE_TASK_FOR_LAYER_H

#include "local-execution/local_ready_to_launch_task.dtg.h"
#include "local-execution/local_task_registry.dtg.h"
#include "pcg/layer_guid_t.dtg.h"
#include "task-spec/runtime_task_invocation.dtg.h"
#include "local-execution/local_atomic_tensor_backing.dtg.h"
#include "task-spec/runtime_arg_config.dtg.h"
#include "local-execution/local_tensor_backing.dtg.h"
#include "task-spec/training_symbolic_computation_graph.dtg.h"
#include "utils/units/milliseconds_t.h"

namespace FlexFlow {

LocalReadyToLaunchTask prepare_runtime_task_invocation(
  RuntimeTaskInvocation const &,
  LocalTensorBacking const &,
  LocalAtomicTensorBacking const &,
  Allocator &,
  RuntimeArgConfig const &);

std::optional<milliseconds_t> execute_forward_for_layer(
  layer_guid_t,
  TrainingSymbolicComputationGraph const &,
  LocalTensorBacking const &,
  LocalAtomicTensorBacking const &,
  Allocator &,
  LocalTaskRegistry const &,
  RuntimeArgConfig const &);

std::optional<milliseconds_t> execute_backward_for_layer(
  layer_guid_t,
  TrainingSymbolicComputationGraph const &,
  LocalTensorBacking const &,
  LocalAtomicTensorBacking const &,
  Allocator &,
  LocalTaskRegistry const &,
  RuntimeArgConfig const &);

} // namespace FlexFlow

#endif
