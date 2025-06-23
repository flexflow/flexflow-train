#include "local-execution/local_tensor_backing.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "pcg/computation_graph.h"
#include "pcg/optimizer_attrs.h"
#include "task-spec/slot_grad_id.dtg.h"
#include "task-spec/training_computation_graph.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/keys.h"
#include "utils/containers/map_values.h"
#include "utils/containers/merge_maps.h"
#include "utils/containers/set_minus.h"
#include "utils/containers/set_of.h"
#include "utils/overload.h"

namespace FlexFlow {

LocalTensorBacking construct_local_tensor_backing(
    std::unordered_map<training_tensor_guid_t, TensorShape> const
        &training_tensor_shapes,
    std::unordered_map<training_tensor_guid_t, GenericTensorAccessorW> const
        &preallocated,
    Allocator &allocator) {

  std::unordered_set<training_tensor_guid_t> to_allocate =
      set_minus(keys(training_tensor_shapes), keys(preallocated));

  std::unordered_map<training_tensor_guid_t, GenericTensorAccessorW> allocated =
      generate_map(to_allocate, [&](training_tensor_guid_t t) {
        TensorShape shape = training_tensor_shapes.at(t);
        return allocator.allocate_tensor(shape);
      });

  std::unordered_map<training_tensor_guid_t, GenericTensorAccessorW>
      backing_for_training_tensor_map =
          merge_disjoint_maps(allocated, preallocated);

  ASSERT(keys(backing_for_training_tensor_map) == keys(training_tensor_shapes));

  return LocalTensorBacking{
      backing_for_training_tensor_map,
  };
}

GenericTensorAccessorW get_accessor_for_training_tensor(
    LocalTensorBacking const &local_tensor_backing,
    training_tensor_guid_t training_tensor) {
  return local_tensor_backing.backing_for_training_tensor_map.at(
      training_tensor);
}

std::unordered_map<tensor_sub_slot_id_t, TensorSlotBacking>
    construct_tensor_slots_backing_for_binding(
        LocalTensorBacking const &local_tensor_backing,
        TaskBinding const &binding) {

  return map_values(
      binding.get_tensor_bindings(), [&](training_tensor_guid_t t) {
        return TensorSlotBacking{
            get_accessor_for_training_tensor(local_tensor_backing, t),
        };
      });
}

} // namespace FlexFlow
