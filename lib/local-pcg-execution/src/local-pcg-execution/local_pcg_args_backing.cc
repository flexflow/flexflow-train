#include "local-pcg-execution/local_pcg_args_backing.h"

namespace FlexFlow {

std::unordered_map<symbolic_layer_guid_t, std::optional<DeviceSpecificPerDeviceOpState>>
  get_op_states_for_machine_space_coord(LocalPcgArgsBacking const &args_backing, MachineSpaceCoordinate const &coord) {
  
  return map_values(
    args_backing.per_device_op_states,
    [&](std::optional<MappedPerDeviceOpStatesGroup> const &m_g) {
      return transform(
        m_g, 
        [&](MappedPerDeviceOpStatesGroup const &g) {
          return g.get_per_device_op_states().at_l(coord); 
        });
    });
}


//
//
// TaskArgumentAccessor
//     get_task_arg_accessor(LocalParallelTensorBacking const &local_parallel_tensor_backing,
//                           RuntimeArgConfig const &runtime_arg_config,
//                           TaskInvocation const &invocation,
//                           Allocator &allocator) {
//   std::unordered_map<training_tensor_slot_id_t, TensorSlotBacking>
//       tensor_slots_backing = construct_tensor_slots_backing_for_binding(
//           local_tensor_backing, invocation.binding);
//
//   std::unordered_map<slot_id_t, ConcreteArgSpec> arg_slots_backing = 
//       construct_arg_slots_backing(invocation.binding, runtime_arg_config);
//
//   return TaskArgumentAccessor::create<LocalTaskArgumentAccessor>(
//       allocator, tensor_slots_backing, arg_slots_backing, );
// }
//
// LocalPcgArgsBacking make_local_pcg_args_backing_for_parallel_computation_graph(
//     LocalTaskRegistry const &task_registry,
//     TrainingParallelComputationGraph const &training_pcg,
//     RuntimeArgConfig const &runtime_arg_config,
//     LocalParallelTensorBacking const &local_parallel_tensor_backing,
//     Allocator &allocator) {
//
//   std::unordered_map<parallel_layer_instance_id, std::optional<DeviceSpecificPerDeviceOpState>>
//       per_device_op_states = generate_map(
//           get_parallel_layers(training_pcg.pcg),
//           [&](parallel_layer_instance_id const &parallel_layer_guid) {
//             return create_per_device_op_state(
//                 task_registry,
//                 local_tensor_backing,
//                 runtime_arg_config,
//                 allocator,
//                 get_training_layer_plus_context(training_computation_graph,
//                                                 layer_guid));
//           });
//
// }


} // namespace FlexFlow
