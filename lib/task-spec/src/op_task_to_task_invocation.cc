#include "task-spec/op_task_to_task_invocation.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "pcg/computation_graph.h"

namespace FlexFlow {

TaskInvocation lower_to_task_invocation(
    OpTaskInvocation const &op_task_invocation,
    layer_guid_t const &layer_guid,
    ComputationGraph const &computation_graph,
    std::unordered_map<tensor_guid_t, gradient_tensor_t> const
        &tensor_gradient_mapping,
    std::optional<DeviceSpecificDeviceStates> const &device_states) {
  TaskBinding binding;
  // tensors
  std::vector<tensor_guid_t> input_tensors =
      get_incoming_inputs(computation_graph, layer_guid);
  std::vector<tensor_guid_t> output_tensors =
      get_outgoing_tensors(computation_graph, layer_guid);
  std::vector<tensor_guid_t> weight_tensors =
      get_incoming_weights(computation_graph, layer_guid);

  for (auto const &tensor_binding :
       op_task_invocation.binding.get_tensor_bindings()) {
    tensor_guid_t tensor_to_bind = [&] {
      OpTensorSpec tensor_binding_spec = tensor_binding.second;
      switch (tensor_binding_spec.role) {
        case TensorRole::INPUT:
          return input_tensors.at(tensor_binding_spec.idx);
        case TensorRole::OUTPUT:
          return output_tensors.at(tensor_binding_spec.idx);
        case TensorRole::WEIGHT:
          return weight_tensors.at(tensor_binding_spec.idx);
        default:
          throw mk_runtime_error(
              fmt::format("Invalid tensor role {}", tensor_binding_spec.role));
      }
    }();

    SlotGradId slot_grad_id = tensor_binding.first;

    if (slot_grad_id.is_grad == IsGrad::NO) {
      binding.bind(slot_grad_id.slot_id, tensor_to_bind);
    } else if (slot_grad_id.is_grad == IsGrad::YES) {
      binding.bind_grad(slot_grad_id.slot_id,
                        tensor_gradient_mapping.at(tensor_to_bind));
    } else {
      throw mk_runtime_error(fmt::format("Invalid value for IsGrad {}",
                                         tensor_binding.first.is_grad));
    }
  }

  // args
  for (auto const &arg_binding :
       op_task_invocation.binding.get_arg_bindings()) {
    if (arg_binding.second.has<OpArgRefSpec>()) {
      ConcreteArgSpec concrete_arg =
          lower_to_concrete_arg_spec(arg_binding.second.get<OpArgRefSpec>(),
                                     computation_graph,
                                     layer_guid,
                                     device_states);
      binding.insert_arg_spec(arg_binding.first, TaskArgSpec{concrete_arg});
    } else if (arg_binding.second.has<RuntimeArgRefSpec>()) {
      binding.insert_arg_spec(
          arg_binding.first,
          TaskArgSpec{arg_binding.second.get<RuntimeArgRefSpec>()});
    } else {
      binding.insert_arg_spec(
          arg_binding.first,
          TaskArgSpec{arg_binding.second.get<ConcreteArgSpec>()});
    }
  }

  return TaskInvocation{op_task_invocation.task_id, binding};
}

ConcreteArgSpec lower_to_concrete_arg_spec(
    OpArgRefSpec const &op_arg_ref_spec,
    ComputationGraph const &cg,
    layer_guid_t const &op_guid,
    std::optional<DeviceSpecificDeviceStates> const &device_states) {
  if (op_arg_ref_spec.holds<DeviceSpecificDeviceStates>()) {
    PerDeviceOpState device_state =
        get_device_state_from_device_specific(device_states.value(), 0);
    return ConcreteArgSpec::create(device_state);
  } else if (op_arg_ref_spec.holds<ParallelTensorShape>()) {
    ParallelTensorShapeRefType index_op_arg_ref =
        op_arg_ref_spec.get_ref_type().get<ParallelTensorShapeRefType>();
    tensor_guid_t input_tensor =
        get_incoming_inputs(cg, op_guid).at(index_op_arg_ref.idx);
    TensorAttrs tensor_attrs = get_tensor_attrs(cg, input_tensor);
    ParallelTensorShape shape = lift_to_parallel(tensor_attrs.shape);
    return ConcreteArgSpec::create(shape);
  } else {
    throw mk_runtime_error("Unhandled op arg ref type");
  }
}

ConcreteArgSpec
    lower_to_concrete_arg_spec(RuntimeArgRefSpec const &runtime_arg_ref_spec,
                               RuntimeArgConfig const &runtime_arg_config) {
  if (runtime_arg_ref_spec.holds<DeviceSpecific<PerDeviceFFHandle>>()) {
    return ConcreteArgSpec::create(*(runtime_arg_config.ff_handle.get(0)));
  } else if (runtime_arg_ref_spec.holds<ProfilingSettings>()) {
    return ConcreteArgSpec::create(runtime_arg_config.profiling_settings);
  } else {
    throw mk_runtime_error("Unhandled runtime arg ref type");
  }
}

} // namespace FlexFlow
