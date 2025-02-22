#include "local-execution/local_tensor_backing.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "pcg/computation_graph.h"
#include "pcg/optimizer_attrs.h"
#include "task-spec/slot_grad_id.dtg.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/keys.h"
#include "utils/overload.h"

namespace FlexFlow {

GenericTensorAccessorW
    get_tensor(LocalTensorBacking const &local_tensor_backing,
               TensorTypeVariant const &tensor_type) {
  return local_tensor_backing.tensor_backings.at(tensor_type);
}

std::unordered_map<tensor_guid_t, std::vector<optimizer_tensor_t>>
    merge_optimizer_mappings(
        std::unordered_map<tensor_guid_t, std::vector<optimizer_tensor_t>> const
            &allocated,
        std::unordered_map<tensor_guid_t, std::vector<optimizer_tensor_t>> const
            &unallocated) {
  std::unordered_map<tensor_guid_t, std::vector<optimizer_tensor_t>>
      merged_maps = allocated;
  for (std::pair<tensor_guid_t, std::vector<optimizer_tensor_t>> const
           &unallocated_optimizer_tensors : unallocated) {
    if (merged_maps.count(unallocated_optimizer_tensors.first)) {
      for (optimizer_tensor_t const &optimizer_tensor :
           unallocated_optimizer_tensors.second) {
        merged_maps[unallocated_optimizer_tensors.first].push_back(
            optimizer_tensor);
      }
    } else {
      merged_maps.insert({unallocated_optimizer_tensors});
    }
  }
  return merged_maps;
}

std::unordered_map<TensorTypeVariant, GenericTensorAccessorW>
    get_tensor_backings(
        std::unordered_map<TensorTypeVariant, GenericTensorAccessorW> const
            &tensor_type_backings,
        std::unordered_map<TensorTypeVariant, TensorShape> const
            &tensor_type_shapes,
        Allocator &allocator) {
  std::unordered_map<TensorTypeVariant, GenericTensorAccessorW>
      all_tensor_backings = tensor_type_backings;

  // allocate new tensors
  for (std::pair<TensorTypeVariant, TensorShape> const &tensor_type_shape :
       tensor_type_shapes) {
    GenericTensorAccessorW tensor_backing =
        allocator.allocate_tensor(tensor_type_shape.second);
    all_tensor_backings.insert({tensor_type_shape.first, tensor_backing});
  }

  return all_tensor_backings;
}

LocalTensorBacking construct_local_tensor_backing(
    AllocatedTensors const &allocated_tensors,
    UnallocatedTensors const &unallocated_tensors,
    Allocator &allocator) {

  std::unordered_map<tensor_guid_t, gradient_tensor_t> merged_gradient_maps =
      allocated_tensors.gradient_mapping;
  merged_gradient_maps.insert(unallocated_tensors.gradient_mapping.begin(),
                              unallocated_tensors.gradient_mapping.end());

  return LocalTensorBacking{
      get_tensor_backings(allocated_tensors.tensor_type_backings,
                          unallocated_tensors.tensor_type_shapes,
                          allocator),
      merged_gradient_maps,
      merge_optimizer_mappings(allocated_tensors.optimizer_mapping,
                               unallocated_tensors.optimizer_mapping)};
}

TensorSlotsBacking construct_tensor_slots_backing(
    LocalTensorBacking const &local_tensor_backing,
    TaskBinding const &binding) {
  TensorSlotsBacking mapping;

  for (std::pair<SlotTensorTypeId, TensorTypeVariant> const &tensor_binding :
       binding.get_tensor_bindings()) {
    mapping.insert({tensor_binding.first,
                    get_tensor(local_tensor_backing, tensor_binding.second)});
  }

  return mapping;
}

} // namespace FlexFlow
