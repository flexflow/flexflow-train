#include "task-spec/lower_op_task_invocation_to_runtime_task_invocation.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "pcg/computation_graph.h"
#include "task-spec/fwb_tensor_slot_id_t.dtg.h"
#include "task-spec/symbolic_layer_tensor_shape_signature.h"
#include "task-spec/symbolic_layer_training_tensor_group_signature_with_shapes.h"
#include "task-spec/symbolic_layer_training_tensor_group_signature.h"
#include "task-spec/training_tensor_slot_id_t.h"
#include "utils/containers/map_values.h"
#include "utils/containers/transform.h"
#include "utils/overload.h"
#include "task-spec/symbolic_layer_training_tensor_group_signature_with_shapes.h"

namespace FlexFlow {

RuntimeTaskInvocation
  lower_op_task_invocation_to_task_invocation(
    OpTaskInvocation const &op_task_invocation,
    symbolic_layer_guid_t symbolic_layer_guid,
    SymbolicLayerTrainingTensorGroupSignatureWithShapes const &layer_signature) {

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

  std::unordered_map<slot_id_t, RuntimeArgSpec> arg_bindings = map_values(
      op_task_invocation.binding.get_arg_bindings(),
      [&](OpArgSpec const &op_arg_spec) -> RuntimeArgSpec {
        return lower_op_arg_spec_to_runtime_arg_spec(op_arg_spec,
                                                     symbolic_layer_guid,
                                                     get_shape_signature(layer_signature));
      });

  return RuntimeTaskInvocation{
      op_task_invocation.task_id,
      RuntimeTaskBinding{
        tensor_bindings,
        arg_bindings,
      },
  };
}

TrainingTensorSlotBinding
    lower_fwb_tensor_binding_to_training_tensor_binding(
                         SymbolicLayerTrainingTensorGroupSignature const &signature,
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

RuntimeArgSpec lower_op_arg_spec_to_runtime_arg_spec(
    OpArgSpec const &op_arg_spec,
    symbolic_layer_guid_t symbolic_layer_guid,
    SymbolicLayerTensorShapeSignature const &op_shape_signature) {
  return op_arg_spec.visit<RuntimeArgSpec>(overload{
      [](ConcreteArgSpec const &concrete_arg_spec) -> RuntimeArgSpec {
        return RuntimeArgSpec{concrete_arg_spec};
      },
      [](RuntimeArgRefSpec const &runtime_arg_ref_spec) -> RuntimeArgSpec {
        return RuntimeArgSpec{runtime_arg_ref_spec};
      },
      [&](OpArgRefSpec const &op_arg_ref_spec) -> RuntimeArgSpec {
        return 
            lower_op_arg_ref_spec_to_runtime_arg_spec(op_arg_ref_spec,
                                       symbolic_layer_guid,
                                       op_shape_signature);
      },
  });
}

RuntimeArgSpec lower_op_arg_ref_spec_to_runtime_arg_spec(
    OpArgRefSpec const &op_arg_ref_spec,
    symbolic_layer_guid_t symbolic_layer_guid,
    SymbolicLayerTensorShapeSignature const &op_signature) {

  OpArgRefType op_arg_ref_type = op_arg_ref_spec.get_ref_type();
  return op_arg_ref_type.visit<RuntimeArgSpec>(overload{
      [&](PerDeviceOpStateRefType const &) -> RuntimeArgSpec {
        return RuntimeArgSpec{
          RuntimeArgRefSpec::create(per_device_op_state_for_layer(symbolic_layer_guid)),
        };
      },
      [&](ParallelTensorShapeRefType const &ref_type) -> RuntimeArgSpec {
        TensorShape tensor_shape = tensor_shape_for_role_and_index(
            /*signature=*/op_signature,
            /*tensor_role=*/ref_type.tensor_role,
            /*index=*/ref_type.idx);
        ParallelTensorShape shape = lift_to_parallel(tensor_shape);
        return RuntimeArgSpec{
          ConcreteArgSpec::create(shape),
        };
      },
  });
}

ConcreteArgSpec
    lower_runtime_arg_ref_spec_to_concrete_arg_spec(RuntimeArgRefSpec const &runtime_arg_ref_spec,
                                                    RuntimeArgConfig const &runtime_arg_config,
                                                    DeviceSpecific<device_handle_t> const &handle,
                                                    std::function<DeviceSpecificPerDeviceOpState(symbolic_layer_guid_t)> const &get_op_state_for_layer) {
  RuntimeArgRefType ref_type = runtime_arg_ref_spec.get_ref_type();

  return ref_type.visit<ConcreteArgSpec>(overload {
    [&](ArgumentlessRuntimeArgRefType argumentless_ref_type) 
      -> ConcreteArgSpec
    {
      return lower_argumentless_arg_ref_to_concrete_arg_spec(
        argumentless_ref_type,
        runtime_arg_config,
        handle);
    },
    [&](PerDeviceOpStateRuntimeArgRefType op_state_ref_type) 
      -> ConcreteArgSpec
    {
      DeviceSpecificPerDeviceOpState op_state = get_op_state_for_layer(op_state_ref_type.layer);

      return ConcreteArgSpec::create(op_state);
    }
  });
}

ConcreteArgSpec lower_argumentless_arg_ref_to_concrete_arg_spec(
    ArgumentlessRuntimeArgRefType ref_type,
    RuntimeArgConfig const &runtime_arg_config,
    DeviceSpecific<device_handle_t> const &handle) {

  switch (ref_type) {
    case ArgumentlessRuntimeArgRefType::FF_HANDLE:
      return ConcreteArgSpec::create(handle);
    case ArgumentlessRuntimeArgRefType::PROFILING_SETTINGS:
      return ConcreteArgSpec::create(runtime_arg_config.profiling_settings);
    case ArgumentlessRuntimeArgRefType::FF_ITERATION_CONFIG:
      PANIC("FF_ITERATION_CONFIG is currently not handled. Please create an "
            "issue or contact the FlexFlow train developers if you need this "
            "feature.");
    case ArgumentlessRuntimeArgRefType::KERNEL_DEVICE_TYPE:
      return ConcreteArgSpec::create(runtime_arg_config.kernel_device_type);
    default:
      PANIC("Unhandled RuntimeArgRefType", ref_type);
  }
}

} // namespace FlexFlow
