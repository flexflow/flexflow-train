#include "local-pcg-execution/execute_tasks_for_parallel_layer.h"
#include "local-execution/local_atomic_tensor_backing.h"
#include "local-pcg-execution/local_parallel_tensor_backing.h"
#include "local-pcg-execution/task_group_execution_times.dtg.h"
#include "task-spec/fwb_op_task_type.h"
#include "task-spec/training_cg_op_attrs_and_signature_with_shapes.dtg.h"

namespace FlexFlow {

std::unordered_map<MachineSpaceCoordinate, LocalReadyToLaunchTask> prepare_parallel_runtime_task_invocations(
  RuntimeTaskInvocation const &runtime_task_invocation,
  LocalParallelTensorBacking const &parallel_tensor_backing,
  LocalAtomicTensorBacking const &atomic_tensor_backing,
  Allocator &allocator,
  RuntimeArgConfig const &runtime_arg_config,
  MappedOperatorTaskGroup const &task_group) {

  std::unordered_set<AtomicTaskInvocation> 
    atomic_task_invocations = 
      lower_parallel_runtime_task_invocation_to_atomic_task_invocation_set(
        parallel_tensor_backing,
        runtime_task_invocation,
        runtime_arg_config,
        task_group);

  return transform(atomic_task_invocations,
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

static std::optional<TaskGroupExecutionTimes> execute_fwb_for_parallel_layer(
  symbolic_layer_guid_t symbolic_layer_guid,
  TrainingSymbolicComputationGraph const &g,
  LocalParallelTensorBacking const &parallel_tensor_backing,
  LocalAtomicTensorBacking const &atomic_tensor_backing,
  Allocator &allocator,
  LocalTaskRegistry const &task_registry,
  RuntimeArgConfig const &runtime_arg_config,
  MappedOperatorTaskGroup const &task_group, 
  FwbOpTaskType fwb_task_type) {
  
  TrainingCgOpAttrsAndSignatureWithShapes attrs_and_signature = 
    get_attrs_and_signature_for_layer(g, symbolic_layer_guid);

  OpTaskType op_task_type = assert_unwrap(
    op_task_type_from_fwb_op_task_type(fwb_task_type));

  std::optional<RuntimeTaskInvocation> maybe_runtime_task_invocation =
    get_runtime_task_invocation_for_layer_and_type(
      symbolic_layer_guid,
      attrs_and_signature,
      op_task_type);

  std::optional<TaskGroupExecutionTimes> execution_times = 
    flatmap(maybe_runtime_task_invocation,
            [&](RuntimeTaskInvocation const &runtime_task_invocation) -> std::optional<milliseconds_t> {
              std::unordered_map<MachineSpaceCoordinate, LocalReadyToLaunchTask> ready = prepare_runtime_task_invocation(
                runtime_task_invocation,
                parallel_tensor_backing,
                atomic_tensor_backing,
                allocator,
                runtime_arg_config);

              
              return call_fwb_task_impl(
                task_group,
                ready.task_id,
                ready.task_arg_accessor);
            });
}

std::optional<TaskGroupExecutionTimes> execute_forward_for_parallel_layer(
  symbolic_layer_guid_t symbolic_layer_guid,
  TrainingSymbolicComputationGraph const &g,
  LocalParallelTensorBacking const &parallel_tensor_backing,
  LocalAtomicTensorBacking const &atomic_tensor_backing,
  Allocator &allocator,
  LocalTaskRegistry const &task_registry,
  RuntimeArgConfig const &runtime_arg_config) {

  return execute_fwb_for_parallel_layer(
    symbolic_layer_guid,
    g,
    parallel_tensor_backing,
    atomic_tensor_backing,
    allocator,
    task_registry,
    runtime_arg_config,
    FwbOpTaskType::FWD);
}

std::optional<TaskGroupExecutionTimes> execute_backward_for_parallel_layer(
  symbolic_layer_guid_t symbolic_layer_guid,
  TrainingSymbolicComputationGraph const &g,
  LocalParallelTensorBacking const &parallel_tensor_backing,
  LocalAtomicTensorBacking const &atomic_tensor_backing,
  Allocator &allocator,
  LocalTaskRegistry const &task_registry,
  RuntimeArgConfig const &runtime_arg_config) {

  return execute_fwb_for_parallel_layer(
    symbolic_layer_guid,
    g,
    parallel_tensor_backing,
    atomic_tensor_backing,
    allocator,
    task_registry,
    runtime_arg_config,
    FwbOpTaskType::BWD);
}




} // namespace FlexFlow
