#include "local-execution/local_args_backing.h"
#include "local-execution/local_task_registry.h"
#include "local-execution/local_tensor_backing.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "task-spec/op_task_to_task_invocation.h"
#include "task-spec/training_computation_graph.h"
#include "task-spec/training_layer_plus_context.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/map_values.h"
#include "utils/overload.h"
#include "utils/containers/try_at.h"
#include "task-spec/task_signature_impl.h"

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

std::optional<DeviceSpecificDeviceStates>
  create_per_device_op_state(LocalTaskRegistry const &task_registry,
                             LocalTensorBacking const &tensor_backing,
                             RuntimeArgConfig const &runtime_arg_config,
                             Allocator &allocator,
                             TrainingLayerPlusContext const &training_layer) {
  if (!registry_contains_task_for_layer(
          task_registry, training_layer.layer_guid, OpTaskType::INIT)) {
    return std::nullopt;
  }
  TaskInvocation invocation =
      lower_to_task_invocation(
        /*op_task_invocation=*/get_init_op_task_invocation(training_layer.layer_attrs.op_attrs),
        /*training_layer=*/training_layer,
        /*device_specific_device_states=*/std::nullopt);

  TaskArgumentAccessor accessor = get_task_arg_accessor(
      tensor_backing,
      runtime_arg_config,
      invocation,
      allocator);
  TaskSignatureAndImpl task_sig_impl =
      task_registry.task_mapping.at(invocation.task_id);
  auto fn = task_sig_impl.impl_function.get<InitOpTaskImplFunction>()
                .function_ptr;
  std::optional<DeviceSpecificDeviceStates> device_state = fn(accessor);
  return device_state;
}

TaskArgumentAccessor
    get_task_arg_accessor(LocalTensorBacking const &local_tensor_backing,
                          RuntimeArgConfig const &runtime_arg_config,
                          TaskInvocation const &invocation,
                          Allocator &allocator) {
  std::unordered_map<tensor_sub_slot_id_t, TensorSlotBacking> tensor_slots_backing =
      construct_tensor_slots_backing_for_binding(local_tensor_backing, invocation.binding);
  std::unordered_map<slot_id_t, ConcreteArgSpec> arg_slots_backing 
    = construct_arg_slots_backing(invocation.binding, runtime_arg_config);
  return TaskArgumentAccessor::create<LocalTaskArgumentAccessor>(
      allocator, tensor_slots_backing, arg_slots_backing);
}

LocalArgsBacking
    make_local_args_backing_for_computation_graph(LocalTaskRegistry const &task_registry,
                            TrainingComputationGraph const &training_computation_graph,
                            RuntimeArgConfig const &runtime_arg_config,
                            LocalTensorBacking const &local_tensor_backing,
                            Allocator &allocator) {
  std::unordered_map<layer_guid_t, std::optional<DeviceSpecificDeviceStates>>
      per_device_op_states = generate_map(
          topological_ordering(training_computation_graph.computation_graph),
          [&](layer_guid_t const &layer_guid) {
            return create_per_device_op_state(
               task_registry,
               local_tensor_backing,
               runtime_arg_config,
               allocator,
               get_training_layer_plus_context(training_computation_graph, layer_guid));
          });

  return LocalArgsBacking{
    runtime_arg_config, 
    per_device_op_states,
  };
}


} // namespace FlexFlow
