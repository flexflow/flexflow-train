#include "local-execution/unallocated_tensors.h"
#include "local-execution/allocated_tensors.h"
#include "pcg/optimizer_attrs.h"

namespace FlexFlow {

UnallocatedTensors generate_unallocated_tensors(
    AllocatedTensors const &allocated_tensors,
    std::unordered_map<tensor_guid_t, TensorAttrs> const &tensor_attrs_mapping,
    GradientTensorSource &gradient_tensor_source) {

  assert(are_allocated_tensors_valid(allocated_tensors, tensor_attrs_mapping));

  std::unordered_map<TensorTypeVariant, TensorShape> tensor_type_shapes;
  std::unordered_map<tensor_guid_t, gradient_tensor_t> gradient_mapping;

  for (std::pair<tensor_guid_t, TensorAttrs> const &tensor_guid_attrs :
       tensor_attrs_mapping) {
    tensor_guid_t tensor_guid = tensor_guid_attrs.first;
    TensorAttrs tensor_attrs = tensor_guid_attrs.second;
    TensorTypeVariant tensor_guid_type = TensorTypeVariant{tensor_guid};
    if (!allocated_tensors.tensor_type_backings.count(tensor_guid_type)) {
      tensor_type_shapes.insert({tensor_guid_type, tensor_attrs.shape});
    }

    if (tensor_attrs.create_grad == CreateGrad::YES &&
        !allocated_tensors.gradient_mapping.count(tensor_guid)) {
      gradient_tensor_t gradient_tensor =
          gradient_tensor_source.new_gradient_tensor();
      tensor_type_shapes.insert(
          {TensorTypeVariant{gradient_tensor}, tensor_attrs.shape});
      gradient_mapping.insert({tensor_guid, gradient_tensor});
    }
  }

  return UnallocatedTensors{tensor_type_shapes, gradient_mapping, {}};
}

UnallocatedTensors generate_unallocated_tensors_with_optimizer(
    AllocatedTensors const &allocated_tensors,
    std::unordered_map<tensor_guid_t, TensorAttrs> const &tensor_attrs_mapping,
    GradientTensorSource &gradient_tensor_source,
    OptimizerTensorSource &optimizer_tensor_source,
    OptimizerAttrs const &optimizer_attrs) {

  UnallocatedTensors unallocated_tensors = generate_unallocated_tensors(
      allocated_tensors, tensor_attrs_mapping, gradient_tensor_source);

  if (!get_num_optimizer_tensors(optimizer_attrs)) {
    return unallocated_tensors;
  }

  std::unordered_map<TensorTypeVariant, TensorShape> tensor_type_shapes =
      unallocated_tensors.tensor_type_shapes;
  std::unordered_map<tensor_guid_t, gradient_tensor_t> gradient_mapping =
      unallocated_tensors.gradient_mapping;
  std::unordered_map<tensor_guid_t, std::vector<optimizer_tensor_t>>
      optimizer_mapping;

  for (std::pair<tensor_guid_t, TensorAttrs> const &tensor_guid_attrs :
       tensor_attrs_mapping) {
    tensor_guid_t tensor_guid = tensor_guid_attrs.first;
    TensorAttrs tensor_attrs = tensor_guid_attrs.second;
    if (tensor_attrs.create_grad == CreateGrad::YES) {
      std::vector<optimizer_tensor_t> optimizer_tensors;

      int num_optimizer_tensors_to_allocate =
          get_num_optimizer_tensors(optimizer_attrs);
      if (allocated_tensors.optimizer_mapping.count(tensor_guid)) {
        num_optimizer_tensors_to_allocate -=
            allocated_tensors.optimizer_mapping.at(tensor_guid).size();
      }
      std::cout << num_optimizer_tensors_to_allocate;

      for (int i = 0; i < num_optimizer_tensors_to_allocate; ++i) {
        optimizer_tensor_t optimizer_tensor =
            optimizer_tensor_source.new_optimizer_tensor();
        optimizer_tensors.push_back(optimizer_tensor);
        tensor_type_shapes.insert(
            {TensorTypeVariant{optimizer_tensor}, tensor_attrs.shape});
      }

      if (num_optimizer_tensors_to_allocate > 0) {
        optimizer_mapping.insert({tensor_guid, optimizer_tensors});
      }
    }
  }

  return UnallocatedTensors{
      tensor_type_shapes, gradient_mapping, optimizer_mapping};
}

} // namespace FlexFlow
