#include "local-execution/local_training_backing.h"
#include "local-execution/loss_functions.h"
#include "local-execution/optimizer.h"
#include "local-execution/task_signature_impl.h"
#include "local-execution/unallocated_tensors.h"
#include "pcg/computation_graph.h"
#include "pcg/optimizer_attrs.h"
#include "task-spec/op_task_to_task_invocation.h"
#include "task-spec/task_invocation.h"
#include "utils/containers/contains.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/get_only.h"
#include "utils/containers/values.h"
#include "utils/exception.h"

namespace FlexFlow {

LocalTrainingBacking::LocalTrainingBacking(
    Allocator &allocator,
    AllocatedTensors const &allocated_tensors,
    GradientTensorSource &gradient_tensor_source,
    ComputationGraph const &computation_graph,
    RuntimeArgConfig const &runtime_arg_config)
    : computation_graph(computation_graph),
      task_registry(
          construct_task_registry(get_layer_attrs_mapping(computation_graph))),
      local_tensor_backing(construct_local_tensor_backing(
          allocated_tensors,
          generate_unallocated_tensors(allocated_tensors,
                                       get_all_tensor_attrs(computation_graph),
                                       gradient_tensor_source),
          allocator)),
      local_args_backing(initialize_args_backing(this->task_registry,
                                                 computation_graph,
                                                 runtime_arg_config,
                                                 this->local_tensor_backing,
                                                 allocator)){};

LocalTrainingBacking::LocalTrainingBacking(
    Allocator &allocator,
    AllocatedTensors const &allocated_tensors,
    GradientTensorSource &gradient_tensor_source,
    OptimizerTensorSource &optimizer_tensor_source,
    ComputationGraph const &computation_graph,
    RuntimeArgConfig const &runtime_arg_config,
    OptimizerAttrs const &optimizer_attrs)
    : computation_graph(computation_graph),
      task_registry(
          construct_task_registry(get_layer_attrs_mapping(computation_graph))),
      local_tensor_backing(construct_local_tensor_backing(
          allocated_tensors,
          generate_unallocated_tensors_with_optimizer(
              allocated_tensors,
              get_all_tensor_attrs(computation_graph),
              gradient_tensor_source,
              optimizer_tensor_source,
              optimizer_attrs),
          allocator)),
      local_args_backing(initialize_args_backing(this->task_registry,
                                                 computation_graph,
                                                 runtime_arg_config,
                                                 this->local_tensor_backing,
                                                 allocator)){};
LocalArgsBacking
    initialize_args_backing(TaskRegistry const &task_registry,
                            ComputationGraph const &cg,
                            RuntimeArgConfig const &runtime_arg_config,
                            LocalTensorBacking const &local_tensor_backing,
                            Allocator &allocator) {
  std::unordered_map<layer_guid_t, DeviceSpecificDeviceStates>
      per_device_op_states;
  for (layer_guid_t const &node : topological_ordering(cg)) {
    if (registry_contains_task_for_layer(
            task_registry, node, OpTaskType::INIT)) {
      ComputationGraphOpAttrs attrs = get_layer_attrs(cg, node).op_attrs;

      TaskInvocation invocation =
          lower_to_task_invocation(init(attrs),
                                   node,
                                   get_incoming_inputs(cg, node),
                                   get_incoming_input_shapes(cg, node),
                                   get_outgoing_tensors(cg, node),
                                   get_incoming_weights(cg, node),
                                   local_tensor_backing.tensor_gradient_mapping,
                                   std::nullopt);
      TaskArgumentAccessor accessor = get_task_arg_accessor(
          local_tensor_backing,
          make_args_backing_with_empty_device_states(runtime_arg_config),
          invocation,
          allocator);
      TaskSignatureAndImpl task_sig_impl =
          task_registry.task_mapping.at(invocation.task_id);
      auto fn = task_sig_impl.impl_function.get<InitOpTaskImplFunction>()
                    .function_ptr;
      DeviceSpecificDeviceStates device_state = fn(accessor);
      per_device_op_states.insert({node, device_state});
    }
  }

  return LocalArgsBacking{runtime_arg_config, per_device_op_states};
}

std::optional<float> call_task_impl(TaskRegistry const &task_registry,
                                    task_id_t const &task_id,
                                    TaskArgumentAccessor const &acc) {
  TaskSignatureAndImpl task_sig_impl = task_registry.task_mapping.at(task_id);
  auto fn =
      task_sig_impl.impl_function.get<FwdBwdOpTaskImplFunction>().function_ptr;
  return fn(acc);
}

std::optional<float>
    execute_forward(LocalTrainingBacking const &local_training_backing,
                    layer_guid_t const &operator_node,
                    Allocator &allocator) {
  if (registry_contains_task_for_layer(local_training_backing.task_registry,
                                       operator_node,
                                       OpTaskType::FWD)) {

    ComputationGraphOpAttrs attrs =
        get_layer_attrs(local_training_backing.computation_graph, operator_node)
            .op_attrs;

    std::optional<DeviceSpecificDeviceStates> device_state =
        get_per_device_op_state_if_exists(
            local_training_backing.local_args_backing, operator_node);

    TaskInvocation invocation = lower_to_task_invocation(
        forward(attrs),
        operator_node,
        get_incoming_inputs(local_training_backing.computation_graph,
                            operator_node),
        get_incoming_input_shapes(local_training_backing.computation_graph,
                                  operator_node),
        get_outgoing_tensors(local_training_backing.computation_graph,
                             operator_node),
        get_incoming_weights(local_training_backing.computation_graph,
                             operator_node),
        local_training_backing.local_tensor_backing.tensor_gradient_mapping,
        device_state);
    TaskArgumentAccessor accessor =
        get_task_arg_accessor(local_training_backing.local_tensor_backing,
                              local_training_backing.local_args_backing,
                              invocation,
                              allocator);
    return call_task_impl(
        local_training_backing.task_registry, invocation.task_id, accessor);
  } else {
    return std::nullopt;
  }
}

void compute_loss(LocalTrainingBacking const &local_training_backing,
                  LossAttrs const &loss_attrs,
                  tensor_guid_t const &logit_tensor,
                  loss_tensor_t const &label_tensor,
                  Allocator &allocator) {
  TaskInvocation loss_invocation = backward(
      loss_attrs,
      logit_tensor,
      local_training_backing.local_tensor_backing.tensor_gradient_mapping.at(
          logit_tensor),
      label_tensor);
  // TODO: https://github.com/flexflow/flexflow-train/issues/1442
  // assert(is_invocation_valid(get_loss_bwd_signature(), loss_invocation));
  TaskArgumentAccessor loss_accessor =
      get_task_arg_accessor(local_training_backing.local_tensor_backing,
                            local_training_backing.local_args_backing,
                            loss_invocation,
                            allocator);
  TaskImplFunction loss_impl_fn = get_loss_bwd_task_impl();
  loss_impl_fn.get<GenericTaskImplFunction>().function_ptr(loss_accessor);
}

std::optional<float>
    execute_backward(LocalTrainingBacking const &local_training_backing,
                     layer_guid_t const &operator_node,
                     Allocator &allocator) {
  if (registry_contains_task_for_layer(local_training_backing.task_registry,
                                       operator_node,
                                       OpTaskType::BWD)) {
    ComputationGraphOpAttrs attrs =
        get_layer_attrs(local_training_backing.computation_graph, operator_node)
            .op_attrs;

    std::optional<DeviceSpecificDeviceStates> device_state =
        get_per_device_op_state_if_exists(
            local_training_backing.local_args_backing, operator_node);
    TaskInvocation invocation = lower_to_task_invocation(
        backward(attrs),
        operator_node,
        get_incoming_inputs(local_training_backing.computation_graph,
                            operator_node),
        get_incoming_input_shapes(local_training_backing.computation_graph,
                                  operator_node),
        get_outgoing_tensors(local_training_backing.computation_graph,
                             operator_node),
        get_incoming_weights(local_training_backing.computation_graph,
                             operator_node),
        local_training_backing.local_tensor_backing.tensor_gradient_mapping,
        device_state);
    TaskArgumentAccessor accessor =
        get_task_arg_accessor(local_training_backing.local_tensor_backing,
                              local_training_backing.local_args_backing,
                              invocation,
                              allocator);
    return call_task_impl(
        local_training_backing.task_registry, invocation.task_id, accessor);
  } else {
    return std::nullopt;
  }
}

void execute_update(LocalTrainingBacking const &local_training_backing,
                    layer_guid_t const &node,
                    OptimizerAttrs const &optimizer_attrs,
                    Allocator &allocator) {
  LayerAttrs layer_attrs =
      get_layer_attrs(local_training_backing.computation_graph, node);
  if (layer_attrs.op_attrs.has<WeightAttrs>()) {
    // get tensors
    tensor_guid_t weight_tensor = get_only(
        get_outgoing_tensors(local_training_backing.computation_graph, node));
    gradient_tensor_t weight_grad_tensor =
        local_training_backing.local_tensor_backing.tensor_gradient_mapping.at(
            weight_tensor);
    std::vector<optimizer_tensor_t> optimizer_buffer_tensors =
        local_training_backing.local_tensor_backing.tensor_optimizer_mapping.at(
            weight_tensor);

    // get invocation
    TaskInvocation invocation = get_update_invocation(optimizer_attrs,
                                                      weight_tensor,
                                                      weight_grad_tensor,
                                                      optimizer_buffer_tensors);

    // TODO: https://github.com/flexflow/flexflow-train/issues/1442
    // assert(is_invocation_valid(get_update_signature(attrs), invocation));

    // execute update
    TaskArgumentAccessor accessor =
        get_task_arg_accessor(local_training_backing.local_tensor_backing,
                              local_training_backing.local_args_backing,
                              invocation,
                              allocator);
    TaskImplFunction update_impl_fn = get_update_task_impl(optimizer_attrs);
    update_impl_fn.get<GenericTaskImplFunction>().function_ptr(accessor);
  }
}

TaskArgumentAccessor
    get_task_arg_accessor(LocalTensorBacking const &local_tensor_backing,
                          LocalArgsBacking const &local_args_backing,
                          TaskInvocation const &invocation,
                          Allocator &allocator) {
  TensorSlotsBacking tensor_slots_backing =
      construct_tensor_slots_backing(local_tensor_backing, invocation.binding);
  ArgSlotsBacking arg_slots_backing = construct_arg_slots_backing(
      invocation.binding, local_args_backing.runtime_arg_config);
  return TaskArgumentAccessor::create<LocalTaskArgumentAccessor>(
      allocator, tensor_slots_backing, arg_slots_backing);
}

} // namespace FlexFlow
