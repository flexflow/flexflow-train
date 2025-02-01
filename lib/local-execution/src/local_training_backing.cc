#include "local-execution/local_training_backing.h"
#include "local-execution/loss_functions.h"
#include "local-execution/op_task_to_task_invocation.h"
#include "local-execution/optimizer.h"
#include "local-execution/task_invocation.h"
#include "local-execution/task_signature_impl.h"
#include "local-execution/tensor_lowering.h"
#include "pcg/computation_graph.h"
#include "pcg/optimizer_attrs.h"
#include "utils/containers/contains.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/get_only.h"
#include "utils/containers/values.h"
#include "utils/exception.h"

namespace FlexFlow {

LocalTrainingBacking::LocalTrainingBacking(
    Allocator const &allocator,
    ComputationGraph const &computation_graph,
    LocalTensorBacking const &local_tensor_backing,
    LocalArgsBacking const &local_args_backing)
    : allocator(allocator), computation_graph(computation_graph),
      task_registry(empty_task_registry()),
      local_tensor_backing(local_tensor_backing),
      local_args_backing(local_args_backing) {
  allocate_all_computation_graph_tensors(this->local_tensor_backing,
                                         this->gradient_tensor_source,
                                         this->computation_graph,
                                         this->allocator);
  register_all_computation_graph_tasks(this->task_registry,
                                       this->computation_graph);
}

DeviceSpecificDeviceStates
    call_init_task_impl(TaskRegistry const &task_registry,
                        task_id_t task_id,
                        TaskArgumentAccessor const &acc) {
  TaskSignatureAndImpl task_sig_impl = task_registry.task_mapping.at(task_id);
  auto fn =
      task_sig_impl.impl_function.get<InitOpTaskImplFunction>().function_ptr;
  return fn(acc);
}

std::optional<float> call_task_impl(TaskRegistry const &task_registry,
                                    task_id_t task_id,
                                    TaskArgumentAccessor acc) {
  TaskSignatureAndImpl task_sig_impl = task_registry.task_mapping.at(task_id);
  auto fn =
      task_sig_impl.impl_function.get<FwdBwdOpTaskImplFunction>().function_ptr;
  return fn(acc);
}

void execute_init(LocalTrainingBacking &local_training_backing,
                  layer_guid_t const &operator_node) {
  if (registry_contains_task_for_layer(local_training_backing.task_registry,
                                       operator_node,
                                       OpTaskType::INIT)) {
    ComputationGraphOpAttrs attrs =
        get_layer_attrs(local_training_backing.computation_graph, operator_node)
            .attrs;

    TaskInvocation invocation =
        lower_to_task_invocation(init(attrs),
                                 operator_node,
                                 local_training_backing.computation_graph,
                                 std::nullopt);
    TaskArgumentAccessor accessor =
        get_task_arg_accessor(local_training_backing.local_tensor_backing,
                              local_training_backing.local_args_backing,
                              invocation);
    DeviceSpecificDeviceStates device_state = call_init_task_impl(
        local_training_backing.task_registry, invocation.task_id, accessor);
    add_per_device_op_state(
        local_training_backing.local_args_backing, operator_node, device_state);
  }
}

std::optional<float>
    execute_forward(LocalTrainingBacking &local_training_backing,
                    layer_guid_t const &operator_node) {
  if (registry_contains_task_for_layer(local_training_backing.task_registry,
                                       operator_node,
                                       OpTaskType::FWD)) {
    ComputationGraphOpAttrs attrs =
        get_layer_attrs(local_training_backing.computation_graph, operator_node)
            .attrs;

    std::optional<DeviceSpecificDeviceStates> device_state =
        get_per_device_op_state_if_exists(
            local_training_backing.local_args_backing, operator_node);
    TaskInvocation invocation =
        lower_to_task_invocation(forward(attrs),
                                 operator_node,
                                 local_training_backing.computation_graph,
                                 device_state);
    TaskArgumentAccessor accessor =
        get_task_arg_accessor(local_training_backing.local_tensor_backing,
                              local_training_backing.local_args_backing,
                              invocation,
                              local_training_backing.allocator);
    return call_task_impl(
        local_training_backing.task_registry, invocation.task_id, accessor);
  } else {
    return std::nullopt;
  }
}

void compute_loss(LocalTrainingBacking const &local_training_backing,
                  LossAttrs const &loss_attrs,
                  tensor_guid_t const &logit_tensor,
                  loss_tensor_t const &label_tensor) {
  TaskInvocation loss_invocation =
      backward(loss_attrs, logit_tensor, label_tensor);
  // TODO: https://github.com/flexflow/flexflow-train/issues/1442
  // assert(is_invocation_valid(get_loss_bwd_signature(), loss_invocation));
  TaskArgumentAccessor loss_accessor =
      get_task_arg_accessor(local_training_backing.local_tensor_backing,
                            local_training_backing.local_args_backing,
                            loss_invocation);
  TaskImplFunction loss_impl_fn = get_loss_bwd_task_impl();
  loss_impl_fn.get<GenericTaskImplFunction>().function_ptr(loss_accessor);
}

std::optional<float>
    execute_backward(LocalTrainingBacking &local_training_backing,
                     layer_guid_t const &operator_node) {
  if (registry_contains_task_for_layer(local_training_backing.task_registry,
                                       operator_node,
                                       OpTaskType::BWD)) {
    ComputationGraphOpAttrs attrs =
        get_layer_attrs(local_training_backing.computation_graph, operator_node)
            .attrs;

    std::optional<DeviceSpecificDeviceStates> device_state =
        get_per_device_op_state_if_exists(
            local_training_backing.local_args_backing, operator_node);
    TaskInvocation invocation =
        lower_to_task_invocation(backward(attrs),
                                 operator_node,
                                 local_training_backing.computation_graph,
                                 device_state);
    TaskArgumentAccessor accessor =
        get_task_arg_accessor(local_training_backing.local_tensor_backing,
                              local_training_backing.local_args_backing,
                              invocation,
                              local_training_backing.allocator);
    return call_task_impl(
        local_training_backing.task_registry, invocation.task_id, accessor);
  } else {
    return std::nullopt;
  }
}

void execute_update(LocalTrainingBacking &local_training_backing,
                    layer_guid_t const &node,
                    OptimizerAttrs const &optimizer_attrs) {
  LayerAttrs layer_attrs =
      get_layer_attrs(local_training_backing.computation_graph, node);
  if (layer_attrs.attrs.has<WeightAttrs>()) {
    // get tensors
    tensor_guid_t weight_tensor = get_only(
        get_outgoing_tensors(local_training_backing.computation_graph, node));
    std::vector<optimizer_tensor_t> optimizer_buffer_tensors =
        local_training_backing.local_tensor_backing.tensor_optimizer_mapping.at(
            weight_tensor);

    // get invocation
    TaskInvocation invocation = get_update_invocation(
        optimizer_attrs, weight_tensor, optimizer_buffer_tensors);

    // TODO: https://github.com/flexflow/flexflow-train/issues/1442
    // assert(is_invocation_valid(get_update_signature(attrs), invocation));

    // execute update
    TaskArgumentAccessor accessor =
        get_task_arg_accessor(local_training_backing.local_tensor_backing,
                              local_training_backing.local_args_backing,
                              invocation,
                              local_training_backing.allocator);
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
