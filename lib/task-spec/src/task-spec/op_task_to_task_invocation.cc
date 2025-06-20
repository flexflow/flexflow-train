#include "task-spec/op_task_to_task_invocation.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "pcg/computation_graph.h"
#include "task-spec/slot_grad_id.dtg.h"
#include "task-spec/training_layer_plus_context.h"

namespace FlexFlow {

TaskInvocation lower_to_task_invocation(
    OpTaskInvocation const &op_task_invocation,
    TrainingLayerPlusContext const &training_layer,
    std::optional<DeviceSpecificDeviceStates> const &device_specific_device_states) {

  return lower_to_task_invocation(
    /*op_task_invocation=*/op_task_invocation,
    /*layer_guid=*/training_layer.layer_guid,
    /*input_tensors=*/get_input_tensors(training_layer),
    /*input_gradient_tensors=*/get_input_grad_tensors(training_layer),
    /*input_tensor_shapes=*/get_input_tensor_shapes(training_layer),
    /*weight_tensors=*/get_weight_tensors(training_layer),
    /*weight_grad_tensors=*/get_weight_grad_tensors(training_layer),
    /*output_tensors=*/get_output_tensors(training_layer),
    /*output_gradient_tensors=*/get_output_grad_tensors(training_layer),
    /*device_specific_device_states=*/device_specific_device_states);
}

TaskInvocation lower_to_task_invocation(
    OpTaskInvocation const &op_task_invocation,
    layer_guid_t const &layer_guid,
    std::vector<forward_tensor_guid_t> const &input_tensors,
    std::vector<gradient_tensor_guid_t> const &input_gradient_tensors,
    std::vector<TensorShape> const &input_tensor_shapes,
    std::vector<forward_tensor_guid_t> const &output_tensors,
    std::vector<gradient_tensor_guid_t> const &output_gradient_tensors,
    std::vector<forward_tensor_guid_t> const &weight_tensors,
    std::vector<gradient_tensor_guid_t> const &weight_gradient_tensors,
    std::optional<DeviceSpecificDeviceStates> const &device_states) {
  TaskBinding binding;

  for (auto const &tensor_binding :
       op_task_invocation.binding.get_tensor_bindings()) {
    auto [tensor_to_bind, gradient_tensor_guid_to_bind] = [&] {
      OpTensorSpec tensor_binding_spec = tensor_binding.second;
      switch (tensor_binding_spec.role) {
        case TensorRole::INPUT:
          return std::pair{
            input_tensors.at(tensor_binding_spec.idx.unwrap_nonnegative()),
            input_gradient_tensors.at(tensor_binding_spec.idx.unwrap_nonnegative()),
          };
        case TensorRole::OUTPUT:
          return std::pair{
            output_tensors.at(tensor_binding_spec.idx.unwrap_nonnegative()),
            output_gradient_tensors.at(tensor_binding_spec.idx.unwrap_nonnegative()),
          };
        case TensorRole::WEIGHT:
          return std::pair{
            weight_tensors.at(tensor_binding_spec.idx.unwrap_nonnegative()),
            weight_gradient_tensors.at(tensor_binding_spec.idx.unwrap_nonnegative()),
          };
        default:
          PANIC("Invalid tensor role", tensor_binding_spec.role);
      }
    }();

    SlotGradId slot_grad_id = tensor_binding.first;

    if (slot_grad_id.is_grad == IsGrad::NO) {
      binding.bind(slot_grad_id.slot_id, tensor_to_bind);
    } else if (slot_grad_id.is_grad == IsGrad::YES) {
      binding.bind_grad(slot_grad_id.slot_id, gradient_tensor_guid_to_bind);
    } else {
      PANIC("Invalid value for IsGrad {}", tensor_binding.first.is_grad);
    }
  }

  // args
  for (auto const &arg_binding :
       op_task_invocation.binding.get_arg_bindings()) {
    if (arg_binding.second.has<OpArgRefSpec>()) {
      ConcreteArgSpec concrete_arg =
          lower_to_concrete_arg_spec(arg_binding.second.get<OpArgRefSpec>(),
                                     input_tensor_shapes,
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
    std::vector<TensorShape> const &input_tensor_shapes,
    layer_guid_t const &op_guid,
    std::optional<DeviceSpecificDeviceStates> const &device_states) {
  if (op_arg_ref_spec.holds<DeviceSpecificDeviceStates>()) {
    PerDeviceOpState device_state =
        get_device_state_from_device_specific(device_states.value(), 0);
    return ConcreteArgSpec::create(device_state);
  } else if (op_arg_ref_spec.holds<ParallelTensorShape>()) {
    ParallelTensorShapeRefType index_op_arg_ref =
        op_arg_ref_spec.get_ref_type().get<ParallelTensorShapeRefType>();
    TensorShape input_tensor_shape =
        input_tensor_shapes.at(index_op_arg_ref.idx.unwrap_nonnegative());
    ParallelTensorShape shape = lift_to_parallel(input_tensor_shape);
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
