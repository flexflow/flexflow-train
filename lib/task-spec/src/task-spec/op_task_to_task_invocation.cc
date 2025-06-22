#include "task-spec/op_task_to_task_invocation.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "pcg/cg_operator_tensor_shape_signature.h"
#include "pcg/computation_graph.h"
#include "task-spec/slot_grad_id.dtg.h"
#include "task-spec/training_layer_plus_context.h"
#include "task-spec/training_layer_tensor_group_signature.h"
#include "utils/overload.h"
#include "utils/containers/map_values.h"
#include "utils/containers/transform.h"

namespace FlexFlow {

TaskInvocation lower_to_task_invocation(
    OpTaskInvocation const &op_task_invocation,
    TrainingLayerPlusContext const &training_layer,
    std::optional<DeviceSpecificDeviceStates> const &device_specific_device_states) {

  std::unordered_map<tensor_sub_slot_id_t, training_tensor_guid_t> tensor_bindings = 
    transform(op_task_invocation.binding.get_tensor_bindings(),
            [&](SlotGradId const &slot_grad_id, OpTensorSpec const &op_tensor_spec) {
              return lower_tensor_binding(
                get_tensor_group_signature(training_layer),
                slot_grad_id,
                op_tensor_spec);
            });

  std::unordered_map<slot_id_t, TaskArgSpec> arg_bindings = 
    map_values(op_task_invocation.binding.get_arg_bindings(),
               [&](OpArgSpec const &op_arg_spec) {
                 return lower_to_task_arg_spec(
                   op_arg_spec,
                   get_cg_op_shape_signature(training_layer),
                   training_layer.layer_guid,
                   device_specific_device_states);
               });
  
  return TaskInvocation{
    op_task_invocation.task_id, 
    TaskBinding{
      tensor_bindings,
      arg_bindings,
    },
  };
}

std::pair<tensor_sub_slot_id_t, training_tensor_guid_t> 
    lower_tensor_binding(TrainingLayerTensorGroupSignature const &signature,
                         SlotGradId const &slot_grad_id,
                         OpTensorSpec const &op_tensor_spec) {
  auto [tensor_to_bind, gradient_tensor_guid_to_bind] = [&] {
    TrainingTensorGroup group = 
      get_training_tensor_group_for_role_and_index(
        signature,
        op_tensor_spec.role,
        op_tensor_spec.idx);

    return std::pair{
      group.forward_tensor,
      group.gradient_tensor,
    };
  }();

  if (slot_grad_id.is_grad == IsGrad::NO) {
    return std::pair{
      tensor_sub_slot_id_t{
        slot_grad_id.slot_id,
        TensorType::FORWARD,
      },
      training_tensor_guid_t{
        tensor_to_bind,
      },
    };
  } else if (slot_grad_id.is_grad == IsGrad::YES) {
    return std::pair{
      tensor_sub_slot_id_t{
        slot_grad_id.slot_id,
        TensorType::GRADIENT,
      },
      training_tensor_guid_t{
        gradient_tensor_guid_to_bind,
      },
    };
  } else {
    PANIC("Invalid value for IsGrad {}", slot_grad_id.is_grad);
  }
}

TaskArgSpec lower_to_task_arg_spec(
   OpArgSpec const &op_arg_spec,
   CGOperatorTensorShapeSignature const &op_shape_signature,
   layer_guid_t const &layer_guid,
   std::optional<DeviceSpecificDeviceStates> const &device_specific_device_states
 ) {
  return op_arg_spec.visit<TaskArgSpec>(overload {
    [](ConcreteArgSpec const &concrete_arg_spec) { 
      return TaskArgSpec{concrete_arg_spec};
    },
    [](RuntimeArgRefSpec const &runtime_arg_ref_spec) {
      return TaskArgSpec{runtime_arg_ref_spec};
    },
    [&](OpArgRefSpec const &op_arg_ref_spec) {
      return TaskArgSpec{
        lower_to_concrete_arg_spec(
          op_arg_ref_spec,
          op_shape_signature,
          layer_guid,
          device_specific_device_states
        ),
      };
    },
  });
}

ConcreteArgSpec lower_to_concrete_arg_spec(
    OpArgRefSpec const &op_arg_ref_spec,
    CGOperatorTensorShapeSignature const &op_signature,
    layer_guid_t const &op_guid,
    std::optional<DeviceSpecificDeviceStates> const &device_states) {

  OpArgRefType op_arg_ref_type = op_arg_ref_spec.get_ref_type();
  return op_arg_ref_type.visit<ConcreteArgSpec>(overload {
    [&](PerDeviceOpStateRefType const &) {
      PerDeviceOpState device_state =
          get_device_state_from_device_specific(device_states.value(), 0);
      return ConcreteArgSpec::create(device_state);
    },
    [&](ParallelTensorShapeRefType const &ref_type) {
      TensorShape tensor_shape = tensor_shape_for_role_and_index( 
        /*signature=*/op_signature,
        /*tensor_role=*/ref_type.tensor_role,
        /*index=*/ref_type.idx);
      ParallelTensorShape shape = lift_to_parallel(tensor_shape);
      return ConcreteArgSpec::create(shape);
    },
  });
}

ConcreteArgSpec
    lower_to_concrete_arg_spec(RuntimeArgRefSpec const &runtime_arg_ref_spec,
                               RuntimeArgConfig const &runtime_arg_config) {
  switch (runtime_arg_ref_spec.get_ref_type()) {
    case RuntimeArgRefType::FF_HANDLE:
      return ConcreteArgSpec::create(*(runtime_arg_config.ff_handle.get(0)));
    case RuntimeArgRefType::PROFILING_SETTINGS:
      return ConcreteArgSpec::create(runtime_arg_config.profiling_settings);
    case RuntimeArgRefType::FF_ITERATION_CONFIG:
      PANIC("FF_ITERATION_CONFIG is currently not handled. Please create an issue or contact the FlexFlow train developers if you need this feature.");
    case RuntimeArgRefType::KERNEL_DEVICE_TYPE:
      return ConcreteArgSpec::create(runtime_arg_config.kernel_device_type);
    default:
      PANIC(fmt::format("Unhandled RuntimeArgRefType {}", runtime_arg_ref_spec.get_ref_type()));
  }
}

} // namespace FlexFlow
