#include "local-pcg-execution/local_parallel_tensor_backing.h"
#include "local-pcg-execution/local_pcg_args_backing.dtg.h"
#include "local-pcg-execution/runtime_atomic_task_shard_binding.dtg.h"
#include "task-spec/device_specific_per_device_op_state.dtg.h"
#include "task-spec/lower_op_task_invocation_to_runtime_task_invocation.h"
#include "utils/containers/map_values.h"
#include "utils/containers/map_values2.h"
#include "utils/containers/try_at.h"
#include "local-pcg-execution/local_pcg_args_backing.h"

namespace FlexFlow {

std::unordered_map<MachineSpaceCoordinate, AtomicTaskInvocation>
  lower_parallel_runtime_task_invocation_to_atomic_task_invocation_group(
    LocalParallelTensorBacking const &parallel_tensor_backing,
    LocalPcgArgsBacking const &parallel_args_backing,
    RuntimeTaskInvocation const &runtime_task_invocation,
    MappedRuntimeTaskGroup const &runtime_task_group) {

  std::unordered_map<MachineSpaceCoordinate, RuntimeAtomicTaskShardBinding>
    shard_bindings = runtime_task_group.get_shard_bindings().as_unordered_map();

  return 
    map_values2(
      shard_bindings,
      [&](MachineSpaceCoordinate const &machine_space_coord, 
          RuntimeAtomicTaskShardBinding const &shard_binding) 
        -> AtomicTaskInvocation 
      {
        return lower_parallel_runtime_task_invocation_to_atomic_task_invocation( 
          parallel_tensor_backing,
          runtime_task_invocation,
          parallel_args_backing.runtime_arg_config,
          get_op_states_for_machine_space_coord(parallel_args_backing, machine_space_coord),
          machine_space_coord,
          shard_binding);
      });
}


AtomicTaskInvocation 
  lower_parallel_runtime_task_invocation_to_atomic_task_invocation(
    LocalParallelTensorBacking const &parallel_tensor_backing,
    RuntimeTaskInvocation const &invocation,
    RuntimeArgConfig const &runtime_arg_config,
    std::unordered_map<symbolic_layer_guid_t, std::optional<DeviceSpecificPerDeviceOpState>> const &per_device_op_states,
    MachineSpaceCoordinate const &machine_space_coord,
    RuntimeAtomicTaskShardBinding const &shard_binding) {

  std::unordered_map<training_tensor_slot_id_t, atomic_training_tensor_guid_t> 
    tensor_bindings = map_values(invocation.binding.get_tensor_bindings(),
                                 [&](symbolic_training_tensor_guid_t t) 
                                   -> atomic_training_tensor_guid_t
                                 {
                                   return parallel_tensor_backing.parallel_tensor_map.at(t);
                                 });

  auto get_op_state_for_layer = [&](symbolic_layer_guid_t l) -> std::optional<DeviceSpecificPerDeviceOpState> {
    return per_device_op_states.at(l); 
  };

  std::unordered_map<slot_id_t, ConcreteArgSpec>
    arg_bindings = map_values(invocation.binding.get_arg_bindings(),
                              [&](RuntimeArgSpec const &arg_spec) -> ConcreteArgSpec
                              {
                                return lower_runtime_arg_ref_spec_to_concrete_arg_spec( 
                                  arg_spec,
                                  runtime_arg_config,
                                  get_op_state_for_layer);
                              });

  return AtomicTaskInvocation{
    invocation.task_id,
    AtomicTaskBinding{
      tensor_bindings,  
      arg_bindings,
    },
  };
}

} // namespace FlexFlow
