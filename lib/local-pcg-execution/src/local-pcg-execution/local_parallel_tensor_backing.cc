#include "local-pcg-execution/local_parallel_tensor_backing.h"
#include "compiler/operator_task_signature.dtg.h"
#include "utils/containers/filtermap_values.h"
#include "utils/containers/try_at.h"

namespace FlexFlow {

LocalParallelTensorBacking construct_local_parallel_tensor_backing(
    std::unordered_map<training_parallel_tensor_guid_t, ParallelTensorShape> const &training_ptensor_shapes,
    std::unordered_map<training_parallel_tensor_guid_t, ParallelTensorAccessorsW> const &preallocated_ptensors,
    Allocator &) {
  
  NOT_IMPLEMENTED();
}

ParallelTensorAccessorsW get_accessors_for_training_ptensor(LocalParallelTensorBacking const &local_parallel_tensor_backing,
                                                            training_parallel_tensor_guid_t tensor) {
  return local_tensor_backing.backing_for_training_tensor_map.at(
      training_tensor);
}

// LocalTensorBacking
//   get_local_tensor_backing_for_device(LocalParallelTensorBacking const &backing,
//                                       MachineSpaceCoordinate const &device) {
//   std::unordered_map<
//     training_parallel_tensor_guid_t, GenericTensorAccessorW
//   > for_device = filtermap_values(backing.backing_for_training_ptensor_map,
//                                   [&](ParallelTensorAccessorsW const &a)
//                                     -> std::optional<GenericTensorAccessorW>
//                                   {
//                                     return try_at(a.shard_map, device);
//                                   });
//
//   return LocalTensorBacking{
//     filtermap_keys(),
//   };
// }

std::unordered_map<training_tensor_slot_id_t, TensorSlotBacking>
  construct_tensor_slots_backing_for_binding_and_task(LocalParallelTensorBacking const &backing,
                                                      TaskBinding const &binding,
                                                      TrainingOperatorTaskSignature const &task) {
  return map_values(
    binding.get_tensor_bindings(),
    [&](training_tensor_guid_t t) {
      return TensorSlotBacking{
        backing.backing_for_training_ptensor_map.at(),
      };
    });
}



} // namespace FlexFlow
