#include "task-spec/lower_op_task_invocation_to_task_invocation.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "pcg/computation_graph.h"
#include "task-spec/fwb_tensor_slot_id_t.dtg.h"
#include "task-spec/symbolic_layer_tensor_shape_signature.h"
#include "task-spec/training_layer_symbolic_tensor_group_signature.h"
#include "task-spec/training_layer_symbolic_tensor_group_signature_with_shapes.h"
#include "task-spec/training_tensor_slot_id_t.h"
#include "utils/containers/map_values.h"
#include "utils/containers/transform.h"
#include "utils/overload.h"
#include "task-spec/training_layer_symbolic_tensor_group_signature_with_shapes.h"

namespace FlexFlow {

TaskInvocation
  lower_op_task_invocation_to_task_invocation(
    OpTaskInvocation const &op_task_invocation,
    TrainingLayerSymbolicTensorGroupSignatureWithShapes const &layer_signature,
    std::optional<DeviceSpecificPerDeviceOpState> const &device_specific_device_states) {

  std::unordered_map<training_tensor_slot_id_t, symbolic_training_tensor_guid_t>
      tensor_bindings =
          transform(op_task_invocation.binding.get_tensor_bindings(),
                    [&](fwb_tensor_slot_id_t const &fwb_slot_id,
                        OpTensorSpec const &op_tensor_spec) {
                      FwbTensorSlotBinding fwb_binding = FwbTensorSlotBinding{
                        fwb_slot_id,
                        op_tensor_spec,
                      };

                      TrainingTensorSlotBinding training_binding = 
                        lower_fwb_tensor_binding_to_training_tensor_binding(
                          drop_shapes_from_signature(layer_signature),
                          fwb_binding);
                      
                      return std::pair{
                        training_binding.slot,
                        training_binding.bound,
                      };
                    });

  std::unordered_map<slot_id_t, TaskArgSpec> arg_bindings = map_values(
      op_task_invocation.binding.get_arg_bindings(),
      [&](OpArgSpec const &op_arg_spec) -> TaskArgSpec {
        return lower_op_arg_spec_to_task_arg_spec(op_arg_spec,
                                                  get_shape_signature(layer_signature),
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

TrainingTensorSlotBinding
    lower_fwb_tensor_binding_to_training_tensor_binding(
                         TrainingLayerSymbolicTensorGroupSignature const &signature,
                         FwbTensorSlotBinding const &fwb_slot_binding) {
  fwb_tensor_slot_id_t fwb_slot_id = fwb_slot_binding.slot;
  OpTensorSpec op_tensor_spec = fwb_slot_binding.bound; 

  SymbolicTrainingTensorGroup group = get_training_tensor_group_for_role_and_index(
      signature, op_tensor_spec.role, op_tensor_spec.idx);

  training_tensor_slot_id_t training_tensor_slot = 
    training_tensor_slot_from_fwb_slot(fwb_slot_id);

  symbolic_training_tensor_guid_t training_tensor = [&]() -> symbolic_training_tensor_guid_t {
    switch (fwb_slot_id.is_grad) {
      case IsGrad::NO:
        return symbolic_training_tensor_guid_t{
          group.forward_tensor,
        };
      case IsGrad::YES:
        return symbolic_training_tensor_guid_t{
          group.gradient_tensor,
        };
      default:
        PANIC("Invalid value for IsGrad {}", fwb_slot_id.is_grad);
    } 
  }();

  return TrainingTensorSlotBinding{
    training_tensor_slot, 
    training_tensor,
  };
}

TaskArgSpec lower_op_arg_spec_to_task_arg_spec(
    OpArgSpec const &op_arg_spec,
    SymbolicLayerTensorShapeSignature const &op_shape_signature,
    std::optional<DeviceSpecificPerDeviceOpState> const
        &device_specific_device_states) {
  return op_arg_spec.visit<TaskArgSpec>(overload{
      [](ConcreteArgSpec const &concrete_arg_spec) {
        return TaskArgSpec{concrete_arg_spec};
      },
      [](RuntimeArgRefSpec const &runtime_arg_ref_spec) {
        return TaskArgSpec{runtime_arg_ref_spec};
      },
      [&](OpArgRefSpec const &op_arg_ref_spec) {
        return TaskArgSpec{
            lower_op_arg_ref_spec_to_concrete_arg_spec(op_arg_ref_spec,
                                       op_shape_signature,
                                       device_specific_device_states),
        };
      },
  });
}

ConcreteArgSpec lower_op_arg_ref_spec_to_concrete_arg_spec(
    OpArgRefSpec const &op_arg_ref_spec,
    SymbolicLayerTensorShapeSignature const &op_signature,
    std::optional<DeviceSpecificPerDeviceOpState> const &device_states) {

  OpArgRefType op_arg_ref_type = op_arg_ref_spec.get_ref_type();
  return op_arg_ref_type.visit<ConcreteArgSpec>(overload{
      [&](PerDeviceOpStateRefType const &) {
        PerDeviceOpState per_device_op_state =
            get_device_state_from_device_specific(device_states.value(), 0);

        return per_device_op_state.visit<ConcreteArgSpec>(overload{
            [&](auto const &x) {
              ASSERT(matches<decltype(x)>(op_arg_ref_spec.get_type_index()));
              return ConcreteArgSpec::create(x);
            },
        });
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
      PANIC("FF_ITERATION_CONFIG is currently not handled. Please create an "
            "issue or contact the FlexFlow train developers if you need this "
            "feature.");
    case RuntimeArgRefType::KERNEL_DEVICE_TYPE:
      return ConcreteArgSpec::create(runtime_arg_config.kernel_device_type);
    default:
      PANIC(fmt::format("Unhandled RuntimeArgRefType {}",
                        runtime_arg_ref_spec.get_ref_type()));
  }
}

} // namespace FlexFlow
