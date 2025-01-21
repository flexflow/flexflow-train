#include "local-execution/local_training_backing.h"
#include "local-execution/loss_functions.h"
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
    RuntimeArgConfig const &runtime_arg_config)
    : allocator(allocator), computation_graph(computation_graph),
      local_args_backing(runtime_arg_config),
      task_registry(empty_task_registry()) {};

void LocalTrainingBacking::register_and_allocate_layer(
    layer_guid_t const &node) {
  ComputationGraphOpAttrs attrs =
      get_layer_attrs(this->computation_graph, node).attrs;
  this->local_tensor_backing.allocate_layer_tensors(
      node, this->computation_graph, this->allocator);
  register_tasks_for_layer(this->task_registry, node, attrs);
}

void LocalTrainingBacking::allocate_layer_optimizer_tensors(
    layer_guid_t const &node, OptimizerAttrs const &optimizer_attrs) {
  ComputationGraphOpAttrs attrs =
      get_layer_attrs(this->computation_graph, node).attrs;
  if (attrs.has<WeightAttrs>()) {
    TaskSignature sig = get_update_signature(optimizer_attrs);
    tensor_guid_t weight_tensor =
        get_only(get_outgoing_tensors(this->computation_graph, node));

    std::vector<optimizer_tensor_t> optimizer_tensors;
    for (TensorTypeSlotSpec const & tensor_type_slot_spec: values(sig.tensor_guid_slots)) {
      optimizer_tensors.push_back(this->optimizer_tensor_source.new_optimizer_tensor());
    }
    this->layer_optimizer_tensor_ids.insert({node, optimizer_tensors});
    this->local_tensor_backing.allocate_optimizer_tensors(
        weight_tensor, optimizer_tensors, this->allocator);
  }
}

DeviceSpecificDeviceStates
    LocalTrainingBacking::call_init_task_impl(task_id_t task_id,
                                              TaskArgumentAccessor const &acc) {
  TaskSignatureAndImpl task_sig_impl =
      this->task_registry.task_mapping.at(task_id);
  auto fn =
      task_sig_impl.impl_function.get<InitOpTaskImplFunction>().function_ptr;
  return fn(acc);
}

std::optional<float>
    LocalTrainingBacking::call_task_impl(task_id_t task_id,
                                         TaskArgumentAccessor acc) {
  TaskSignatureAndImpl task_sig_impl =
      this->task_registry.task_mapping.at(task_id);
  auto fn =
      task_sig_impl.impl_function.get<FwdBwdOpTaskImplFunction>().function_ptr;
  return fn(acc);
}

void LocalTrainingBacking::execute_init(layer_guid_t const &operator_node) {
  if (registry_contains_task_for_layer(
          this->task_registry, operator_node, OpTaskType::INIT)) {
    ComputationGraphOpAttrs attrs =
        get_layer_attrs(this->computation_graph, operator_node).attrs;

    TaskInvocation invocation = this->lower_to_task_invocation(init(attrs));
    TaskArgumentAccessor accessor =
        this->get_task_arg_accessor(invocation);
    DeviceSpecificDeviceStates device_state =
        this->call_init_task_impl(invocation.task_id, accessor);
    this->local_args_backing.add_per_device_op_state(operator_node,
                                                      device_state);
  }
}

std::optional<float>
    LocalTrainingBacking::execute_forward(layer_guid_t const &operator_node) {
  if (registry_contains_task_for_layer(
          this->task_registry, operator_node, OpTaskType::FWD)) {
    ComputationGraphOpAttrs attrs =
        get_layer_attrs(this->computation_graph, operator_node).attrs;

    TaskInvocation invocation = this->lower_to_task_invocation(forward(attrs));
    TaskArgumentAccessor accessor =
        this->get_task_arg_accessor(invocation);
    return this->call_task_impl(invocation.task_id, accessor);
  } else {
    return std::nullopt;
  }
}

void LocalTrainingBacking::compute_loss(LossAttrs const &loss_attrs,
                                        tensor_guid_t const &logit_tensor,
                                        loss_tensor_t const &label_tensor) {
  TaskInvocation loss_invocation =
      backward(loss_attrs, logit_tensor, label_tensor);
  // TODO: https://github.com/flexflow/flexflow-train/issues/1442
  // assert(is_invocation_valid(get_loss_bwd_signature(), loss_invocation));
  TaskArgumentAccessor loss_accessor =
      this->get_task_arg_accessor(loss_invocation);
  TaskImplFunction loss_impl_fn = get_loss_bwd_task_impl();
  loss_impl_fn.get<GenericTaskImplFunction>().function_ptr(loss_accessor);
}

std::optional<float>
    LocalTrainingBacking::execute_backward(layer_guid_t const &operator_node) {
  if (registry_contains_task_for_layer(
          this->task_registry, operator_node, OpTaskType::BWD)) {
    ComputationGraphOpAttrs attrs =
        get_layer_attrs(this->computation_graph, operator_node).attrs;

    TaskInvocation invocation = this->lower_to_task_invocation(backward(attrs));
    TaskArgumentAccessor accessor =
        this->get_task_arg_accessor(invocation);
    return this->call_task_impl(invocation.task_id, accessor);
  } else {
    return std::nullopt;
  }
}

void LocalTrainingBacking::execute_update(
    layer_guid_t const &node, OptimizerAttrs const &optimizer_attrs) {
  LayerAttrs layer_attrs = get_layer_attrs(this->computation_graph, node);
  if (layer_attrs.attrs.has<WeightAttrs>()) {
    // get tensors
    tensor_guid_t weight_tensor = get_only(get_outgoing_tensors(this->computation_graph, node));
    std::vector<optimizer_tensor_t> optimizer_buffer_tensors = this->layer_optimizer_tensor_ids.at(node);

    // get invocation
    TaskInvocation invocation = get_update_invocation(
        optimizer_attrs, weight_tensor, optimizer_buffer_tensors);

    // TODO: https://github.com/flexflow/flexflow-train/issues/1442
    // assert(is_invocation_valid(get_update_signature(attrs), invocation));

    // execute update
    TaskArgumentAccessor accessor =
        this->get_task_arg_accessor(invocation);
    TaskImplFunction update_impl_fn = get_update_task_impl(optimizer_attrs);
    update_impl_fn.get<GenericTaskImplFunction>().function_ptr(accessor);
  }
}

TaskArgumentAccessor LocalTrainingBacking::get_task_arg_accessor(
    TaskInvocation const &invocation) const {
  TensorSlotsBacking tensor_slots_backing =
      this->local_tensor_backing.construct_tensor_slots_backing(
          invocation.binding);
  ArgSlotsBacking arg_slots_backing =
      this->local_args_backing.construct_arg_slots_backing(invocation.binding);
  return TaskArgumentAccessor::create<LocalTaskArgumentAccessor>(
      this->allocator, tensor_slots_backing, arg_slots_backing);
}

TaskInvocation LocalTrainingBacking::lower_to_task_invocation(OpTaskInvocation const & op_task_invocation, layer_guid_t const & layer_guid) const {
  TaskBinding binding;
  // tensors
  for (auto const & tensor_binding: op_task_invocation.binding.get_tensor_bindings()) {
    tensor_guid_t tensor_to_bind = [&] {
      switch (tensor_binding.second.role) {
        case TensorRole::INPUT:
          return get_incoming_inputs(this->computation_graph, layer_guid).at(tensor_binding.second.idx);
        case TensorRole::OUTPUT:
          return get_outgoing_tensors(this->computation_graph, layer_guid).at(tensor_binding.second.idx);
        case TensorRole::WEIGHT:
          return get_incoming_weights(this->computation_graph, layer_guid).at(tensor_binding.second.idx);
        default:
          throw mk_runtime_error(fmt::format("Invalid tensor role {}", tensor_binding.second.role));
      }
    }(); 

    if (tensor_binding.first.is_grad == IsGrad::NO) {
      binding.bind(tensor_binding.first.slot_id, tensor_to_bind);
    } else if (tensor_binding.first.is_grad == IsGrad::YES) {
      binding.bind_grad(tensor_binding.first.slot_id, tensor_to_bind);
    } else {
      throw mk_runtime_error(fmt::format("Invalid value for IsGrad {}", tensor_binding.first.is_grad));
    }
  }

  // args
  for (auto const & arg_binding: op_task_invocation.binding.get_arg_bindings()) {
    if (arg_binding.second.has<OpArgRefSpec>()) {
      ConcreteArgSpec concrete_arg = this->local_args_backing.lower_to_concrete_arg_spec(arg_binding.second.get<OpArgRefSpec>(), this->computation_graph, layer_guid);
      binding.insert_arg_spec(arg_binding.first, TaskArgSpec{concrete_arg});
    } else if (arg_binding.second.has<RuntimeArgRefSpec>()) {
      binding.insert_arg_spec(arg_binding.first, TaskArgSpec{arg_binding.second.get<RuntimeArgRefSpec>()});
    } else {
      binding.insert_arg_spec(arg_binding.first, TaskArgSpec{arg_binding.second.get<ConcreteArgSpec>()});
    }
  }

  return TaskInvocation{op_task_invocation.task_id, binding};
}

} // namespace FlexFlow
