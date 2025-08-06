#include "local-execution/local_training_backing.h"
#include "local-execution/local_args_backing.h"
#include "pcg/computation_graph.h"
#include "pcg/optimizer_attrs.h"
#include "task-spec/loss_functions.h"
#include "task-spec/op_task_to_task_invocation.h"
#include "task-spec/optimizer.h"
#include "task-spec/task_invocation.h"
#include "task-spec/task_signature_impl.h"
#include "task-spec/training_computation_graph.h"
#include "utils/containers/contains.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/get_only.h"
#include "utils/containers/is_subseteq_of.h"
#include "utils/containers/keys.h"
#include "utils/containers/values.h"
#include "utils/exception.h"

namespace FlexFlow {

LocalTrainingBacking make_local_training_backing_for_computation_graph(
    Allocator &allocator,
    std::unordered_map<training_tensor_guid_t, GenericTensorAccessorW> const
        &preallocated,
    TrainingComputationGraph const &training_computation_graph,
    RuntimeArgConfig const &runtime_arg_config,
    OptimizerAttrs const &optimizer_attrs) {

  ASSERT(is_subseteq_of(
      keys(preallocated),
      keys(get_all_training_tensor_shapes(training_computation_graph))));

  LocalTaskRegistry local_task_registry =
      construct_local_task_registry_for_layers(get_layer_attrs_mapping(
          training_computation_graph.computation_graph));

  LocalTensorBacking local_tensor_backing = construct_local_tensor_backing(
      get_all_training_tensor_shapes(training_computation_graph),
      preallocated,
      allocator);

  LocalArgsBacking local_args_backing =
      make_local_args_backing_for_computation_graph(local_task_registry,
                                                    training_computation_graph,
                                                    runtime_arg_config,
                                                    local_tensor_backing,
                                                    allocator);

  return LocalTrainingBacking{
      /*computation_graph=*/training_computation_graph,
      /*local_task_registry=*/local_task_registry,
      /*local_tensor_backing=*/local_tensor_backing,
      /*local_args_backing=*/local_args_backing,
  };
}

std::optional<milliseconds_t>
    execute_forward(LocalTaskRegistry const &local_task_registry,
                    LocalTensorBacking const &local_tensor_backing,
                    LocalArgsBacking const &local_args_backing,
                    TrainingLayerPlusContext const &training_layer,
                    Allocator &allocator) {

  std::optional maybe_registered_task = try_get_registered_task(
      local_task_registry, training_layer.layer_guid, OpTaskType::FWD);

  ASSERT(maybe_registered_task.has_value());

  registered_task_t registered_task = maybe_registered_task.value();
  if (registered_task.is_noop_task()) {
    return std::nullopt;
  }

  std::optional<DeviceSpecificDeviceStates> device_state =
      get_per_device_op_state_if_exists(local_args_backing,
                                        training_layer.layer_guid);

  TaskInvocation invocation = lower_to_task_invocation(
      /*op_task_invocation=*/get_forward_op_task_invocation(
          training_layer.layer_attrs.op_attrs),
      /*training_layer=*/training_layer,
      /*device_specific_device_states=*/device_state);

  TaskArgumentAccessor accessor =
      get_task_arg_accessor(local_tensor_backing,
                            local_args_backing.runtime_arg_config,
                            invocation,
                            allocator);

  return call_task_impl(local_task_registry, invocation.task_id, accessor);
}

void compute_loss(LocalTrainingBacking const &local_training_backing,
                  LossAttrs const &loss_attrs,
                  Allocator &allocator) {

  TrainingComputationGraph training_cg =
      local_training_backing.training_computation_graph;
  tensor_guid_t logit_tensor = training_cg.logit_tensor;
  loss_tensor_guid_t label_tensor = training_cg.label_tensor;

  TaskInvocation loss_invocation = backward(
      loss_attrs,
      get_forward_tensor_guid_for_tensor_guid(training_cg, logit_tensor),
      get_gradient_tensor_guid_for_tensor_guid(training_cg, logit_tensor),
      label_tensor);
  // TODO: https://github.com/flexflow/flexflow-train/issues/1442
  // assert(is_invocation_valid(get_loss_bwd_signature(), loss_invocation));
  TaskArgumentAccessor loss_accessor = get_task_arg_accessor(
      local_training_backing.local_tensor_backing,
      local_training_backing.local_args_backing.runtime_arg_config,
      loss_invocation,
      allocator);
  TaskImplFunction loss_impl_fn = get_loss_bwd_task_impl();
  loss_impl_fn.get<GenericTaskImplFunction>().function_ptr(loss_accessor);
}

std::optional<milliseconds_t>
    execute_backward(LocalTaskRegistry const &local_task_registry,
                     LocalTensorBacking const &local_tensor_backing,
                     LocalArgsBacking const &local_args_backing,
                     TrainingLayerPlusContext const &training_layer,
                     Allocator &allocator) {

  std::optional maybe_registered_task = try_get_registered_task(
      local_task_registry, training_layer.layer_guid, OpTaskType::BWD);

  ASSERT(maybe_registered_task.has_value());

  registered_task_t registered_task = maybe_registered_task.value();
  if (registered_task.is_noop_task()) {
    return std::nullopt;
  }

  std::optional<DeviceSpecificDeviceStates> device_state =
      get_per_device_op_state_if_exists(local_args_backing,
                                        training_layer.layer_guid);
  TaskInvocation invocation = lower_to_task_invocation(
      get_backward_op_task_invocation(training_layer.layer_attrs.op_attrs),
      training_layer,
      device_state);
  TaskArgumentAccessor accessor =
      get_task_arg_accessor(local_tensor_backing,
                            local_args_backing.runtime_arg_config,
                            invocation,
                            allocator);
  return call_task_impl(local_task_registry, invocation.task_id, accessor);
}

void execute_update(LocalTrainingBacking const &local_training_backing,
                    layer_guid_t const &layer_guid,
                    OptimizerAttrs const &optimizer_attrs,
                    Allocator &allocator) {
  TrainingLayerPlusContext training_layer = get_training_layer_plus_context(
      local_training_backing.training_computation_graph, layer_guid);

  if (training_layer.layer_attrs.op_attrs.has<WeightAttrs>()) {
    TrainingTensorGroupWithAttrs weight_tensor_group =
        get_only(training_layer.output_tensor_groups);

    TaskInvocation invocation =
        get_update_invocation(optimizer_attrs,
                              weight_tensor_group.forward_tensor,
                              weight_tensor_group.gradient_tensor,
                              weight_tensor_group.optimizer_tensors);

    // TODO: https://github.com/flexflow/flexflow-train/issues/1442
    // assert(is_invocation_valid(get_update_signature(attrs), invocation));

    TaskArgumentAccessor accessor = get_task_arg_accessor(
        local_training_backing.local_tensor_backing,
        local_training_backing.local_args_backing.runtime_arg_config,
        invocation,
        allocator);
    TaskImplFunction update_impl_fn = get_update_task_impl(optimizer_attrs);
    update_impl_fn.get<GenericTaskImplFunction>().function_ptr(accessor);
  }
}

} // namespace FlexFlow
