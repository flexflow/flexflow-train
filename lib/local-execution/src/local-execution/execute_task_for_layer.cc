#include "local-execution/execute_task_for_layer.h"
#include "local-execution/local_atomic_tensor_backing.h"
#include "local-execution/local_ready_to_launch_task.dtg.h"
#include "local-execution/local_task_registry.h"
#include "local-execution/local_tensor_backing.h"
#include "task-spec/fwb_op_task_type.h"
#include "task-spec/runtime_task_invocation.dtg.h"
#include "task-spec/training_symbolic_computation_graph.h"
#include "utils/containers/flatmap.h"

namespace FlexFlow {

LocalReadyToLaunchTask prepare_runtime_task_invocation(
  RuntimeTaskInvocation const &runtime_task_invocation,
  LocalTensorBacking const &local_tensor_backing,
  LocalAtomicTensorBacking const &local_atomic_tensor_backing,
  Allocator &allocator,
  RuntimeArgConfig const &runtime_arg_config) {

  AtomicTaskInvocation atomic_task_invocation = 
    lower_local_runtime_task_invocation_to_atomic_task_invocation(
      local_tensor_backing,
      runtime_task_invocation,
      runtime_arg_config);

  TaskArgumentAccessor task_arg_accessor = 
    get_task_arg_accessor_for_atomic_task_invocation(
      local_atomic_tensor_backing,
      atomic_task_invocation,
      allocator);

  return LocalReadyToLaunchTask{
    atomic_task_invocation.task_id,
    task_arg_accessor,
  };
}

std::optional<DeviceSpecificPerDeviceOpState> execute_init_for_layer(
  layer_guid_t symbolic_layer_guid,
  TrainingSymbolicComputationGraph const &graph,
  LocalTensorBacking const &tensor_backing,
  LocalAtomicTensorBacking const &atomic_tensor_backing,
  Allocator &allocator,
  LocalTaskRegistry const &task_registry,
  RuntimeArgConfig const &runtime_arg_config) {

  TrainingCgOpAttrsAndSignatureWithShapes attrs_and_signature = 
    get_attrs_and_signature_for_layer(g, symbolic_layer_guid);

  std::optional<RuntimeTaskInvocation> maybe_runtime_task_invocation =
    get_init_runtime_task_invocation_for_layer(
      symbolic_layer_guid,
      attrs_and_signature);

  std::optional<DeviceSpecificPerDeviceOpState> per_device_op_state = 
    flatmap(maybe_runtime_task_invocation,
            [&](RuntimeTaskInvocation const &runtime_task_invocation) -> std::optional<DeviceSpecificPerDeviceOpState> {
              LocalReadyToLaunchTask ready = prepare_runtime_task_invocation(
                runtime_task_invocation,
                tensor_backing,
                atomic_tensor_backing,
                allocator,
                runtime_arg_config);

              return call_init_task_impl(
                task_registry,
                ready.task_id,
                ready.task_arg_accessor);
            });

  return per_device_op_state;
}

static std::optional<milliseconds_t> execute_fwb_for_layer(
  layer_guid_t symbolic_layer_guid,
  TrainingSymbolicComputationGraph const &g,
  LocalTensorBacking const &local_tensor_backing,
  LocalAtomicTensorBacking const &local_atomic_tensor_backing,
  Allocator &allocator,
  LocalTaskRegistry const &local_task_registry,
  RuntimeArgConfig const &runtime_arg_config,
  FwbOpTaskType task_type) {

  TrainingCgOpAttrsAndSignatureWithShapes attrs_and_signature = 
    get_attrs_and_signature_for_layer(g, symbolic_layer_guid);

  OpTaskType op_task_type = assert_unwrap(
    op_task_type_from_fwb_op_task_type(task_type));

  std::optional<RuntimeTaskInvocation> maybe_runtime_task_invocation =
    get_runtime_task_invocation_for_layer_and_type(
      symbolic_layer_guid,
      attrs_and_signature,
      op_task_type);

  std::optional<milliseconds_t> elapsed_time = 
    flatmap(maybe_runtime_task_invocation,
            [&](RuntimeTaskInvocation const &runtime_task_invocation) -> std::optional<milliseconds_t> {
              LocalReadyToLaunchTask ready = prepare_runtime_task_invocation(
                runtime_task_invocation,
                local_tensor_backing,
                local_atomic_tensor_backing,
                allocator,
                runtime_arg_config);

              return call_fwb_task_impl(
                local_task_registry,
                ready.task_id,
                ready.task_arg_accessor);
            });
}

std::optional<milliseconds_t> execute_forward_for_layer(
  layer_guid_t layer,
  TrainingSymbolicComputationGraph const &graph,
  LocalTensorBacking const &tensor_backing,
  LocalAtomicTensorBacking const &atomic_tensor_backing,
  Allocator &allocator,
  LocalTaskRegistry const &task_registry,
  RuntimeArgConfig const &runtime_arg_config) {

  return execute_fwb_for_layer(layer, graph, tensor_backing, atomic_tensor_backing, allocator, task_registry, runtime_arg_config, FwbOpTaskType::FWD);
}

std::optional<milliseconds_t> execute_backward_for_layer(
  layer_guid_t,
  TrainingSymbolicComputationGraph const &,
  LocalTensorBacking const &,
  LocalAtomicTensorBacking const &,
  Allocator &,
  LocalTaskRegistry const &,
  RuntimeArgConfig const &);

  return execute_fwb_for_layer(layer, graph, tensor_backing, atomic_tensor_backing, allocator, task_registry, runtime_arg_config, FwbOpTaskType::FWD);
} // namespace FlexFlow

