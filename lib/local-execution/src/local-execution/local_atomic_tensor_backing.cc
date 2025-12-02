#include "local-execution/local_atomic_tensor_backing.h"
#include "local-execution/local_task_argument_accessor.h"
#include "utils/containers/map_values.h"

namespace FlexFlow {

std::unordered_map<training_tensor_slot_id_t, TensorSlotBacking>
    construct_tensor_slots_backing_for_binding(LocalAtomicTensorBacking const &tensor_backing,
                                               AtomicTaskBinding const &binding) {
  return map_values(
      binding.tensor_bindings,
      [&](atomic_training_tensor_guid_t t) -> TensorSlotBacking {
        return TensorSlotBacking{
          tensor_backing.accessor_from_atomic_tensor_map.at(t),
        };
      });
}

TaskArgumentAccessor get_task_arg_accessor_for_atomic_task_invocation(
  LocalAtomicTensorBacking const &local_tensor_backing,
  AtomicTaskBinding const &atomic_task_binding,
  Allocator &allocator) {

  std::unordered_map<training_tensor_slot_id_t, TensorSlotBacking>
      tensor_slots_backing = construct_tensor_slots_backing_for_binding(
          local_tensor_backing, atomic_task_binding);

  std::unordered_map<arg_slot_id_t, ConcreteArgSpec> arg_slots_backing =
    atomic_task_binding.arg_bindings;

  return TaskArgumentAccessor::create<LocalTaskArgumentAccessor>(
      allocator, tensor_slots_backing, arg_slots_backing, 0);
}


} // namespace FlexFlow
