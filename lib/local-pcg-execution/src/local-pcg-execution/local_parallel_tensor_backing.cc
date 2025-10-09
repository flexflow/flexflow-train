#include "local-pcg-execution/local_parallel_tensor_backing.h"
#include "compiler/operator_task_signature.dtg.h"
#include "utils/containers/map_values2.h"
#include "utils/containers/try_at.h"

namespace FlexFlow {

std::unordered_map<MachineSpaceCoordinate, AtomicTaskInvocation>
  lower_parallel_runtime_task_invocation_to_atomic_task_invocation_group(
    LocalParallelTensorBacking const &parallel_tensor_backing,
    RuntimeTaskInvocation const &runtime_task_invocation,
    RuntimeArgConfig const &runtime_arg_config,
    MappedOperatorTaskGroup const &op_task_group) {

  std::unordered_map<MachineSpaceCoordinate, OperatorTaskSignature>
    signature_map = op_task_group.get_task_signatures().as_unordered_map();

  return 
    map_values2(
      signature_map,
      [&](MachineSpaceCoordinate const &machine_space_coord, 
          OperatorTaskSignature const &op_task_signature) 
        -> AtomicTaskInvocation 
      {
        return lower_parallel_runtime_task_invocation_to_atomic_task_invocation( 
          parallel_tensor_backing,
          runtime_task_invocation,
          runtime_arg_config,
          machine_space_coord,
          op_task_signature);
      });
}


AtomicTaskInvocation 
  lower_parallel_runtime_task_invocation_to_atomic_task_invocation(
    LocalParallelTensorBacking const &parallel_tensor_backing,
    RuntimeTaskInvocation const &invocation,
    RuntimeArgConfig const &runtime_arg_config,
    MachineSpaceCoordinate const &machine_space_coord,
    OperatorTaskSignature const &op_task_signature) {

  std::unordered_map<training_tensor_slot_id_t, atomic_training_tensor_guid_t> 
    tensor_bindings = map_values(invocation.get_tensor_bindings(),
                                 [&](symbolic_training_tensor_guid_t t) 
                                   -> atomic_training_tensor_guid_t
                                 {
                                   return parallel_tensor_backing.at(t);
                                 })

  return AtomicTaskInvocation{
    invocation.task_id,
    AtomicTaskBinding{
        
    },
  };
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
