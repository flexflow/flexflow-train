#include "local-execution/local_args_backing.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "task-spec/op_task_to_task_invocation.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/map_values.h"
#include "utils/overload.h"

namespace FlexFlow {

LocalArgsBacking make_args_backing_with_empty_device_states(
    RuntimeArgConfig const &runtime_arg_config) {
  return LocalArgsBacking{runtime_arg_config, {}};
}

LocalArgsBacking::LocalArgsBacking(
    RuntimeArgConfig const &runtime_arg_config,
    std::unordered_map<layer_guid_t, DeviceSpecificDeviceStates> const
        &device_states)
    : runtime_arg_config(runtime_arg_config),
      per_device_op_states(device_states){};

std::optional<DeviceSpecificDeviceStates> get_per_device_op_state_if_exists(
    LocalArgsBacking const &local_args_backing,
    layer_guid_t const &layer_guid) {
  if (contains_key(local_args_backing.per_device_op_states, layer_guid)) {
    return local_args_backing.per_device_op_states.at(layer_guid);
  } else {
    return std::nullopt;
  }
}

ArgSlotsBacking
    construct_arg_slots_backing(TaskBinding const &binding,
                                RuntimeArgConfig const &runtime_arg_config) {
  return map_values(
      binding.get_arg_bindings(), [&](TaskArgSpec const &arg_binding) {
        return arg_binding.template visit<ConcreteArgSpec>(
            overload{[&](RuntimeArgRefSpec const &s) {
                       return lower_to_concrete_arg_spec(s, runtime_arg_config);
                     },
                     [](ConcreteArgSpec const &s) { return s; }});
      });
  ;
}

} // namespace FlexFlow
