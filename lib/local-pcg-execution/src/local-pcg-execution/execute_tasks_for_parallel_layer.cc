#include "local-pcg-execution/execute_tasks_for_parallel_layer.h"
#include "local-execution/local_atomic_tensor_backing.h"
#include "local-execution/local_task_registry.h"
#include "local-pcg-execution/local_parallel_tensor_backing.h"
#include "local-pcg-execution/task_group_execution_times.dtg.h"
#include "task-spec/fwb_op_task_type.h"
#include "task-spec/symbolic/training_symbolic_computation_graph.h"
#include "utils/containers/flatmap.h"
#include "utils/containers/lift_optional_through_map.h"
#include "utils/containers/map_values.h"
#include "utils/containers/values.h"
#include "utils/containers/all_of.h"

namespace FlexFlow {

std::unordered_map<MachineSpaceCoordinate, LocalReadyToLaunchTask> prepare_parallel_runtime_task_invocations(
  RuntimeTaskInvocation const &runtime_task_invocation,
  LocalParallelTensorBacking const &parallel_tensor_backing,
  LocalAtomicTensorBacking const &atomic_tensor_backing,
  Allocator &allocator,
  RuntimeArgConfig const &runtime_arg_config,
  MappedRuntimeTaskGroup const &task_group) {

  std::unordered_map<MachineSpaceCoordinate, AtomicTaskInvocation> 
    atomic_task_invocations = 
      lower_parallel_runtime_task_invocation_to_atomic_task_invocation_group(
        parallel_tensor_backing,
        runtime_task_invocation,
        runtime_arg_config,
        task_group);

  return map_values(atomic_task_invocations,
                    [&](AtomicTaskInvocation const &atomic_task_invocation)
                      -> LocalReadyToLaunchTask
                    {
                      TaskArgumentAccessor task_arg_accessor = 
                        get_task_arg_accessor_for_atomic_task_invocation(
                          atomic_tensor_backing,
                          atomic_task_invocation,
                          allocator);

                      return LocalReadyToLaunchTask{
                        atomic_task_invocation.task_id,
                        task_arg_accessor,
                      };
                    });
}

  std::optional<MappedPerDeviceOpStatesGroup> execute_init_for_parallel_layer(
    symbolic_layer_guid_t symbolic_layer_guid,
    TrainingSymbolicComputationGraph const &g,
    LocalParallelTensorBacking const &parallel_tensor_backing,
    LocalAtomicTensorBacking const &atomic_tensor_backing,
    Allocator &allocator,
    LocalTaskRegistry const &task_registry,
    RuntimeArgConfig const &runtime_arg_config,
    MappedRuntimeTaskGroup const &task_group) {

  SymbolicCgOpAttrsAndTrainingSignatureWithShapes attrs_and_signature = 
    get_attrs_and_signature_for_layer(g, symbolic_layer_guid);

  RuntimeTaskInvocation runtime_task_invocation = ({
    std::optional<RuntimeTaskInvocation> maybe_runtime_task_invocation =
      get_init_runtime_task_invocation_for_layer(
        symbolic_layer_guid,
        attrs_and_signature);
    if (!maybe_runtime_task_invocation.has_value()) {
      return std::nullopt;
    }
    maybe_runtime_task_invocation.value();
  });

  std::unordered_map<MachineSpaceCoordinate, LocalReadyToLaunchTask> 
    prepared_tasks = prepare_parallel_runtime_task_invocations(
      runtime_task_invocation,
      parallel_tensor_backing,
      atomic_tensor_backing,
      allocator,
      runtime_arg_config,
      task_group);
      
  std::unordered_map<MachineSpaceCoordinate, std::optional<DeviceSpecificPerDeviceOpState>> op_state_by_shard =
    map_values(prepared_tasks,
               [&](LocalReadyToLaunchTask const &prepared_task) -> std::optional<DeviceSpecificPerDeviceOpState> {
                 return call_init_task_impl(
                   task_registry,
                   prepared_task.task_id,
                   prepared_task.task_arg_accessor);
               });

  return transform(
    lift_optional_through_map(op_state_by_shard),
    [](std::unordered_map<MachineSpaceCoordinate, DeviceSpecificPerDeviceOpState> const &m) {
      return MappedPerDeviceOpStatesGroup{m}; 
    });
}


static std::optional<TaskGroupExecutionTimes> execute_fwb_for_parallel_layer(
  symbolic_layer_guid_t symbolic_layer_guid,
  TrainingSymbolicComputationGraph const &g,
  LocalParallelTensorBacking const &parallel_tensor_backing,
  LocalAtomicTensorBacking const &atomic_tensor_backing,
  Allocator &allocator,
  LocalTaskRegistry const &task_registry,
  RuntimeArgConfig const &runtime_arg_config,
  MappedRuntimeTaskGroup const &task_group, 
  FwbOpTaskType fwb_task_type) {
  
  SymbolicCgOpAttrsAndTrainingSignatureWithShapes attrs_and_signature = 
    get_attrs_and_signature_for_layer(g, symbolic_layer_guid);

  OpTaskType op_task_type = assert_unwrap(
    op_task_type_from_fwb_op_task_type(fwb_task_type));

  RuntimeTaskInvocation runtime_task_invocation = ({
    std::optional<RuntimeTaskInvocation> maybe_runtime_task_invocation =
      get_runtime_task_invocation_for_layer_and_type(
        symbolic_layer_guid,
        attrs_and_signature,
        op_task_type);
    if (!maybe_runtime_task_invocation.has_value()) {
      return std::nullopt;
    }
    maybe_runtime_task_invocation.value();
  });

  std::unordered_map<MachineSpaceCoordinate, LocalReadyToLaunchTask> 
    prepared_tasks = prepare_parallel_runtime_task_invocations(
      runtime_task_invocation,
      parallel_tensor_backing,
      atomic_tensor_backing,
      allocator,
      runtime_arg_config,
      task_group);
      

  std::unordered_map<MachineSpaceCoordinate, std::optional<milliseconds_t>> timing_by_shard =
    map_values(prepared_tasks,
               [&](LocalReadyToLaunchTask const &prepared_task) -> std::optional<milliseconds_t> {
                 return call_fwb_task_impl(
                   task_registry,
                   prepared_task.task_id,
                   prepared_task.task_arg_accessor);
               });

  return transform(
    lift_optional_through_map(timing_by_shard),
    [](std::unordered_map<MachineSpaceCoordinate, milliseconds_t> const &m) {
      return TaskGroupExecutionTimes{m}; 
    });
}

std::optional<TaskGroupExecutionTimes> execute_forward_for_parallel_layer(
  symbolic_layer_guid_t symbolic_layer_guid,
  TrainingSymbolicComputationGraph const &g,
  LocalParallelTensorBacking const &parallel_tensor_backing,
  LocalAtomicTensorBacking const &atomic_tensor_backing,
  Allocator &allocator,
  LocalTaskRegistry const &task_registry,
  RuntimeArgConfig const &runtime_arg_config,
  MappedRuntimeTaskGroup const &task_group) {

  return execute_fwb_for_parallel_layer(
    symbolic_layer_guid,
    g,
    parallel_tensor_backing,
    atomic_tensor_backing,
    allocator,
    task_registry,
    runtime_arg_config,
    task_group,
    FwbOpTaskType::FWD);
}

std::optional<TaskGroupExecutionTimes> execute_backward_for_parallel_layer(
  symbolic_layer_guid_t symbolic_layer_guid,
  TrainingSymbolicComputationGraph const &g,
  LocalParallelTensorBacking const &parallel_tensor_backing,
  LocalAtomicTensorBacking const &atomic_tensor_backing,
  Allocator &allocator,
  LocalTaskRegistry const &task_registry,
  RuntimeArgConfig const &runtime_arg_config,
  MappedRuntimeTaskGroup const &task_group) {

  return execute_fwb_for_parallel_layer(
    symbolic_layer_guid,
    g,
    parallel_tensor_backing,
    atomic_tensor_backing,
    allocator,
    task_registry,
    runtime_arg_config,
    task_group,
    FwbOpTaskType::BWD);
}

} // namespace FlexFlow
