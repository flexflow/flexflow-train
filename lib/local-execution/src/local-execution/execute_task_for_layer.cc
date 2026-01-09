#if 0 // FIXME (Elliott): fix execute task

#include "local-execution/execute_task_for_layer.h"
#include "local-execution/local_task_registry.h"
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
          local_tensor_backing, runtime_task_invocation, runtime_arg_config);

  TaskArgumentAccessor task_arg_accessor =
      get_task_arg_accessor_for_atomic_task_invocation(
          local_atomic_tensor_backing, atomic_task_invocation, allocator);

  return LocalReadyToLaunchTask{
      atomic_task_invocation.task_id,
      task_arg_accessor,
  };
}

std::optional<DeviceSpecificPerDeviceOpState> execute_init_for_layer(
    symbolic_layer_guid_t symbolic_layer_guid,
    TrainingSymbolicComputationGraph const &g,
    LocalTensorBacking const &tensor_backing,
    LocalAtomicTensorBacking const &atomic_tensor_backing,
    Allocator &allocator,
    LocalTaskRegistry const &task_registry,
    RuntimeArgConfig const &runtime_arg_config) {

  SymbolicCgOpAttrsAndTrainingSignatureWithShapes attrs_and_signature =
      get_attrs_and_signature_for_layer(g, symbolic_layer_guid);

  RuntimeTaskInvocation runtime_task_invocation = ({
    std::optional<RuntimeTaskInvocation> maybe_runtime_task_invocation =
        get_init_runtime_task_invocation_for_layer(symbolic_layer_guid,
                                                   attrs_and_signature);
    if (!maybe_runtime_task_invocation.has_value()) {
      return std::nullopt;
    }
    maybe_runtime_task_invocation.value();
  });

  LocalReadyToLaunchTask prepared_task =
      prepare_runtime_task_invocation(runtime_task_invocation,
                                      tensor_backing,
                                      atomic_tensor_backing,
                                      allocator,
                                      runtime_arg_config);

  std::optional<DeviceSpecificPerDeviceOpState> per_device_op_state =
      call_init_task_impl(task_registry,
                          prepared_task.task_id,
                          prepared_task.task_arg_accessor);

  return per_device_op_state;
}

static std::optional<milliseconds_t> execute_fwb_for_layer(
    symbolic_layer_guid_t symbolic_layer_guid,
    SymbolicCgOpAttrsAndTrainingSignatureWithShapes const &attrs_and_signature,
    LocalTensorBacking const &local_tensor_backing,
    LocalAtomicTensorBacking const &local_atomic_tensor_backing,
    Allocator &allocator,
    LocalTaskRegistry const &local_task_registry,
    RuntimeArgConfig const &runtime_arg_config,
    FwbOpTaskType task_type) {

  OpTaskType op_task_type =
      assert_unwrap(op_task_type_from_fwb_op_task_type(task_type));

  RuntimeTaskInvocation runtime_task_invocation = ({
    std::optional<RuntimeTaskInvocation> maybe_runtime_task_invocation =
        get_runtime_task_invocation_for_layer_and_type(
            symbolic_layer_guid, attrs_and_signature, op_task_type);
    if (!maybe_runtime_task_invocation.has_value()) {
      return std::nullopt;
    }
    maybe_runtime_task_invocation.value();
  });

  task_id_t task_id = runtime_task_invocation.task_id;

  RuntimeTaskBinding runtime_task_binding = runtime_task_invocation.binding;

  AtomicTaskBinding atomic_task_binding =
      lower_local_runtime_task_binding_to_atomic_task_binding(
          local_tensor_backing, runtime_task_binding, runtime_arg_config);

  TaskArgumentAccessor task_arg_accessor =
      get_task_arg_accessor_for_atomic_task_binding(
          local_atomic_tensor_backing, atomic_task_binding, allocator);

  std::optional<milliseconds_t> execution_time =
      call_fwb_task_impl(local_task_registry, task_id, task_arg_accessor);

  return execution_time;
}

std::optional<milliseconds_t> execute_forward_for_layer(
    symbolic_layer_guid_t layer,
    SymbolicCgOpAttrsAndTrainingSignatureWithShapes const &attrs_and_signature,
    LocalTensorBacking const &tensor_backing,
    LocalAtomicTensorBacking const &atomic_tensor_backing,
    Allocator &allocator,
    LocalTaskRegistry const &task_registry,
    RuntimeArgConfig const &runtime_arg_config) {

  return execute_fwb_for_layer(layer,
                               attrs_and_signature,
                               tensor_backing,
                               atomic_tensor_backing,
                               allocator,
                               task_registry,
                               runtime_arg_config,
                               FwbOpTaskType::FWD);
}

std::optional<milliseconds_t> execute_backward_for_layer(
    symbolic_layer_guid_t layer,
    SymbolicCgOpAttrsAndTrainingSignatureWithShapes const &attrs_and_signature,
    LocalTensorBacking const &tensor_backing,
    LocalAtomicTensorBacking const &atomic_tensor_backing,
    Allocator &allocator,
    LocalTaskRegistry const &task_registry,
    RuntimeArgConfig const &runtime_arg_config) {

  return execute_fwb_for_layer(layer,
                               attrs_and_signature,
                               tensor_backing,
                               atomic_tensor_backing,
                               allocator,
                               task_registry,
                               runtime_arg_config,
                               FwbOpTaskType::BWD);
}

void execute_compute_loss(LossAttrs const &loss_attrs,
                          symbolic_forward_tensor_guid_t logit_fwd_tensor,
                          symbolic_gradient_tensor_guid_t logit_grad_tensor,
                          symbolic_loss_tensor_guid_t loss_tensor,
                          LocalTensorBacking const &tensor_backing,
                          LocalAtomicTensorBacking const &atomic_tensor_backing,
                          Allocator &allocator,
                          LocalTaskRegistry const &task_registry,
                          RuntimeArgConfig const &runtime_arg_config) {

  RuntimeTaskInvocation invocation = get_compute_loss_runtime_task_invocation(
      loss_attrs, logit_fwd_tensor, logit_grad_tensor, loss_tensor);

  LocalReadyToLaunchTask prepared_task =
      prepare_runtime_task_invocation(invocation,
                                      tensor_backing,
                                      atomic_tensor_backing,
                                      allocator,
                                      runtime_arg_config);

  call_generic_task_impl(
      task_registry, prepared_task.task_id, prepared_task.task_arg_accessor);
}

void execute_update_for_layer(
    symbolic_layer_guid_t symbolic_layer_guid,
    TrainingSymbolicComputationGraph const &graph,
    LocalTensorBacking const &tensor_backing,
    LocalAtomicTensorBacking const &atomic_tensor_backing,
    OptimizerAttrs const &optimizer_attrs,
    Allocator &allocator,
    LocalTaskRegistry const &task_registry,
    RuntimeArgConfig const &runtime_arg_config) {

  SymbolicTrainingLayerAttrsPlusContext attrs_plus_context =
      get_symbolic_training_layer_attrs_plus_context(graph,
                                                     symbolic_layer_guid);

  RuntimeTaskInvocation invocation = ({
    std::optional<RuntimeTaskInvocation> maybe_invocation =
        get_update_runtime_task_invocation_for_layer(attrs_plus_context,
                                                     optimizer_attrs);
    if (!maybe_invocation.has_value()) {
      return;
    }
    maybe_invocation.value();
  });

  LocalReadyToLaunchTask prepared_task =
      prepare_runtime_task_invocation(invocation,
                                      tensor_backing,
                                      atomic_tensor_backing,
                                      allocator,
                                      runtime_arg_config);

  call_generic_task_impl(
      task_registry, prepared_task.task_id, prepared_task.task_arg_accessor);
}

std::unordered_map<layer_guid_t, std::optional<milliseconds_t>>
    execute_forward_pass(
        TrainingSymbolicComputationGraphFromCgConversion const &training_cg,
        LocalTensorBacking const &local_tensor_backing,
        LocalAtomicTensorBacking const &local_atomic_tensor_backing,
        Allocator &allocator,
        LocalTaskRegistry const &local_task_registry,
        RuntimeArgConfig const &runtime_arg_config) {
  std::unordered_map<layer_guid_t, std::optional<milliseconds_t>>
      per_layer_elapsed_time;

  for (symbolic_layer_guid_t symbolic_layer_guid :
       symbolic_cg_topological_ordering(
           training_cg.training_symbolic_computation_graph)) {

    std::optional<milliseconds_t> elapsed_time = execute_forward_for_layer(
        symbolic_layer_guid,
        training_cg.training_symbolic_computation_graph,
        local_tensor_backing,
        local_atomic_tensor_backing,
        allocator,
        local_task_registry,
        runtime_arg_config);

    layer_guid_t layer_guid =
        training_cg.layer_mapping.at_r(symbolic_layer_guid);
    per_layer_elapsed_time.insert({layer_guid, elapsed_time});
  }

  return per_layer_elapsed_time;
}

std::unordered_map<layer_guid_t, std::optional<milliseconds_t>>
    execute_backward_pass(
        TrainingSymbolicComputationGraphFromCgConversion const &training_cg,
        LocalTensorBacking const &local_tensor_backing,
        LocalAtomicTensorBacking const &local_atomic_tensor_backing,
        Allocator &allocator,
        LocalTaskRegistry const &local_task_registry,
        RuntimeArgConfig const &runtime_arg_config) {
  std::unordered_map<layer_guid_t, std::optional<milliseconds_t>>
      per_layer_elapsed_time;

  for (symbolic_layer_guid_t symbolic_layer_guid :
       reversed(symbolic_cg_topological_ordering(
           training_cg.training_symbolic_computation_graph))) {

    std::optional<milliseconds_t> elapsed_time = execute_backward_for_layer(
        symbolic_layer_guid,
        training_cg.training_symbolic_computation_graph,
        local_tensor_backing,
        local_atomic_tensor_backing,
        allocator,
        local_task_registry,
        runtime_arg_config);

    layer_guid_t layer_guid =
        training_cg.layer_mapping.at_r(symbolic_layer_guid);
    per_layer_elapsed_time.insert({layer_guid, elapsed_time});
  }

  return per_layer_elapsed_time;
}

} // namespace FlexFlow

#endif
