#ifndef _FLEXFLOW_LIB_LOCAL_PCG_EXECUTION_INCLUDE_LOCAL_PCG_EXECUTION_EXECUTE_TASKS_FOR_PARALLEL_LAYER_H
#define _FLEXFLOW_LIB_LOCAL_PCG_EXECUTION_INCLUDE_LOCAL_PCG_EXECUTION_EXECUTE_TASKS_FOR_PARALLEL_LAYER_H

#include "compiler/mapped_operator_task_group.h"
#include "local-execution/local_atomic_tensor_backing.dtg.h"
#include "local-execution/local_ready_to_launch_task.dtg.h"
#include "local-execution/local_task_registry.dtg.h"
#include "local-pcg-execution/local_parallel_tensor_backing.dtg.h"
#include "local-pcg-execution/mapped_per_device_op_states_group.h"
#include "local-pcg-execution/mapped_runtime_task_group.h"
#include "local-pcg-execution/task_group_execution_times.dtg.h"
#include "task-spec/runtime_arg_config.dtg.h"
#include "task-spec/runtime_task_invocation.dtg.h"
#include "task-spec/training_symbolic_computation_graph.dtg.h"

namespace FlexFlow {

std::unordered_map<MachineSpaceCoordinate, LocalReadyToLaunchTask> prepare_parallel_runtime_task_invocations(
  RuntimeTaskInvocation const &,
  LocalParallelTensorBacking const &,
  LocalAtomicTensorBacking const &,
  Allocator &,
  RuntimeArgConfig const &,
  MappedRuntimeTaskGroup const &);

std::optional<MappedPerDeviceOpStatesGroup> execute_init_for_parallel_layer(
  symbolic_layer_guid_t,
  TrainingSymbolicComputationGraph const &,
  LocalParallelTensorBacking const &,
  LocalAtomicTensorBacking const &,
  Allocator &,
  LocalTaskRegistry const &,
  RuntimeArgConfig const &,
  MappedRuntimeTaskGroup const &);

std::optional<TaskGroupExecutionTimes> execute_forward_for_parallel_layer(
  symbolic_layer_guid_t,
  TrainingSymbolicComputationGraph const &,
  LocalParallelTensorBacking const &,
  LocalAtomicTensorBacking const &,
  Allocator &,
  LocalTaskRegistry const &,
  RuntimeArgConfig const &,
  MappedRuntimeTaskGroup const &);

std::optional<TaskGroupExecutionTimes> execute_forward_for_parallel_layer(
  symbolic_layer_guid_t,
  TrainingSymbolicComputationGraph const &,
  LocalParallelTensorBacking const &,
  LocalAtomicTensorBacking const &,
  Allocator &,
  LocalTaskRegistry const &,
  RuntimeArgConfig const &,
  MappedRuntimeTaskGroup const &);



} // namespace FlexFlow

#endif
