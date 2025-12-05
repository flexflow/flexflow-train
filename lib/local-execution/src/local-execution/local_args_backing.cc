#include "local-execution/local_args_backing.h"
#include "local-execution/local_task_registry.h"
#include "local-execution/local_tensor_backing.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "task-spec/op_task_to_task_invocation.h"
#include "task-spec/task_signature_impl.h"
#include "task-spec/training_computation_graph.h"
#include "task-spec/training_layer_plus_context.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/map_values.h"
#include "utils/containers/try_at.h"
#include "utils/overload.h"

namespace FlexFlow {

std::optional<DeviceSpecificDeviceStates> get_per_device_op_state_if_exists(
    LocalArgsBacking const &local_args_backing,
    layer_guid_t const &layer_guid) {

  return local_args_backing.per_device_op_states.at(layer_guid);
}

std::unordered_map<slot_id_t, ConcreteArgSpec>
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

TaskArgumentAccessor
    get_task_arg_accessor(LocalTensorBacking const &local_tensor_backing,
                          RuntimeArgConfig const &runtime_arg_config,
                          TaskInvocation const &invocation,
                          Allocator &allocator) {
  std::unordered_map<tensor_sub_slot_id_t, TensorSlotBacking>
      tensor_slots_backing = construct_tensor_slots_backing_for_binding(
          local_tensor_backing, invocation.binding);
  std::unordered_map<slot_id_t, ConcreteArgSpec> arg_slots_backing =
      construct_arg_slots_backing(invocation.binding, runtime_arg_config);
  return TaskArgumentAccessor::create<LocalTaskArgumentAccessor>(
      allocator, tensor_slots_backing, arg_slots_backing);
}

LocalArgsBacking make_local_args_backing_for_computation_graph(
    RuntimeArgConfig const &runtime_arg_config,
    std::unordered_map<layer_guid_t, std::optional<DeviceSpecificDeviceStates>> const &
        per_device_op_states) {
  return LocalArgsBacking{
      runtime_arg_config,
      per_device_op_states,
  };
}

} // namespace FlexFlow
