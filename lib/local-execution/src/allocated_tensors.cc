#include "local-execution/allocated_tensors.h"
#include "pcg/optimizer_attrs.h"
#include "utils/containers/keys.h"
#include "utils/containers/set_union.h"

namespace FlexFlow {

bool is_allocated_tensor_backing_valid(
    TensorTypeVariant const &tensor_type,
    std::unordered_map<TensorTypeVariant, GenericTensorAccessorW> const
        &allocated_tensor_backings,
    ArrayShape const &expected_shape) {
  if (allocated_tensor_backings.count(tensor_type)) {
    GenericTensorAccessorW tensor_backing =
        allocated_tensor_backings.at(tensor_type);
    if (expected_shape == tensor_backing.shape) {
      return true;
    }
  }
  return false;
};

bool are_allocated_forward_tensors_valid(
    AllocatedTensors const &allocated_tensors,
    ComputationGraph const &computation_graph) {
  std::unordered_set<tensor_guid_t> all_tensor_guids =
      set_union(keys(allocated_tensors.gradient_mapping),
                keys(allocated_tensors.optimizer_mapping));
  for (tensor_guid_t const &tensor_guid : all_tensor_guids) {
    TensorAttrs expected_tensor_attrs =
        get_tensor_attrs(computation_graph, tensor_guid);
    if (!is_allocated_tensor_backing_valid(
            TensorTypeVariant{tensor_guid},
            allocated_tensors.tensor_type_backings,
            ArrayShape{expected_tensor_attrs.shape})) {
      return false;
    }
  }
  return true;
}

bool are_allocated_gradient_tensors_valid(
    AllocatedTensors const &allocated_tensors,
    ComputationGraph const &computation_graph) {
  std::unordered_set<TensorTypeVariant>
      tensors_in_mappings; // will check whether any dangling gradient tensors
                           // were allocated

  for (std::pair<tensor_guid_t, gradient_tensor_t> const &tensor_to_grad :
       allocated_tensors.gradient_mapping) {
    TensorAttrs expected_tensor_attrs =
        get_tensor_attrs(computation_graph, tensor_to_grad.first);
    if (expected_tensor_attrs.create_gradients == CreateGrad::NO) {
      return false;
    }

    ArrayShape tensor_guid_array_shape =
        allocated_tensors.tensor_type_backings
            .at(TensorTypeVariant{tensor_to_grad.first})
            .shape;
    TensorTypeVariant gradient_tensor =
        TensorTypeVariant{tensor_to_grad.second};
    if (is_allocated_tensor_backing_valid(
            gradient_tensor,
            allocated_tensors.tensor_type_backings,
            tensor_guid_array_shape)) {
      tensors_in_mappings.insert(gradient_tensor);
    } else {
      return false;
    }
  }

  for (TensorTypeVariant const &tensor_type :
       keys(allocated_tensors.tensor_type_backings)) {
    if (tensor_type.has<gradient_tensor_t>()) {
      if (!tensors_in_mappings.count(tensor_type)) {
        return false;
      }
    }
  }
  return true;
}

bool are_allocated_optimizer_tensors_valid(
    AllocatedTensors const &allocated_tensors,
    ComputationGraph const &computation_graph) {
  std::unordered_set<TensorTypeVariant>
      tensors_in_mappings; // will check whether any dangling optimizer tensors
                           // were allocated

  for (std::pair<tensor_guid_t, std::vector<optimizer_tensor_t>> const
           &tensor_to_optimizers : allocated_tensors.optimizer_mapping) {
    TensorAttrs expected_tensor_attrs =
        get_tensor_attrs(computation_graph, tensor_to_optimizers.first);
    if (expected_tensor_attrs.create_gradients == CreateGrad::NO) {
      return false;
    }

    ArrayShape tensor_guid_array_shape =
        allocated_tensors.tensor_type_backings
            .at(TensorTypeVariant{tensor_to_optimizers.first})
            .shape;
    for (optimizer_tensor_t const &optimizer_tensor :
         tensor_to_optimizers.second) {
      if (is_allocated_tensor_backing_valid(
              TensorTypeVariant{optimizer_tensor},
              allocated_tensors.tensor_type_backings,
              tensor_guid_array_shape)) {
        tensors_in_mappings.insert(TensorTypeVariant{optimizer_tensor});
      } else {
        return false;
      }
    }
  }

  for (TensorTypeVariant const &tensor_type :
       keys(allocated_tensors.tensor_type_backings)) {
    if (tensor_type.has<optimizer_tensor_t>()) {
      if (!tensors_in_mappings.count(tensor_type)) {
        return false;
      }
    }
  }

  return true;
}

} // namespace FlexFlow
