#include "task-spec/lower_op_task_invocation_to_runtime_task_invocation.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "pcg/computation_graph.h"
#include "task-spec/fwb_tensor_slot_id_t.dtg.h"
#include "task-spec/symbolic/symbolic_layer_tensor_shape_signature.h"
#include "task-spec/symbolic/symbolic_layer_training_tensor_group_signature_with_shapes.h"
#include "task-spec/symbolic/symbolic_layer_training_tensor_group_signature.h"
#include "task-spec/training_tensor_slot_id_t.h"
#include "utils/containers/map_values.h"
#include "utils/containers/transform.h"
#include "utils/overload.h"
#include "task-spec/symbolic/symbolic_layer_training_tensor_group_signature_with_shapes.h"

namespace FlexFlow {

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
