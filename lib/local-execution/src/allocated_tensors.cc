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
    std::unordered_map<tensor_guid_t, TensorAttrs> const &tensor_attrs) {

  std::unordered_set<tensor_guid_t> all_tensor_guids = transform(
      keys(filter_keys(
          allocated_tensors.tensor_type_backings,
          [&](TensorTypeVariant const &k) { return k.has<tensor_guid_t>(); })),
      [&](TensorTypeVariant const &t) { return t.get<tensor_guid_t>(); });

  for (tensor_guid_t const &tensor_guid : all_tensor_guids) {
    if (tensor_attrs.count(tensor_guid)) {
      if (!is_allocated_tensor_backing_valid(
              TensorTypeVariant{tensor_guid},
              allocated_tensors.tensor_type_backings,
              array_shape_from_tensor_shape(tensor_attrs.at(tensor_guid).shape))) {
        return false;
      }
    } else {
      return false;
    }
  }
  return true;
}

bool are_allocated_gradient_tensors_valid(
    AllocatedTensors const &allocated_tensors,
    std::unordered_map<tensor_guid_t, TensorAttrs> const &tensor_attrs) {
  std::unordered_set<TensorTypeVariant>
      tensors_in_mappings; // will check for dangling gradient tensors

  for (std::pair<tensor_guid_t, gradient_tensor_t> const &tensor_to_grad :
       allocated_tensors.gradient_mapping) {
    if (tensor_attrs.count(tensor_to_grad.first)) {
      if (tensor_attrs.at(tensor_to_grad.first).create_grad == CreateGrad::NO) {
        return false;
      }

      ArrayShape tensor_guid_array_shape =
          array_shape_from_tensor_shape(tensor_attrs.at(tensor_to_grad.first).shape);
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
    std::unordered_map<tensor_guid_t, TensorAttrs> const &tensor_attrs) {
  std::unordered_set<TensorTypeVariant>
      tensors_in_mappings; // will check for dangling optimizer tensors

  for (std::pair<tensor_guid_t, std::vector<optimizer_tensor_t>> const
           &tensor_to_optimizers : allocated_tensors.optimizer_mapping) {
    if (tensor_attrs.count(tensor_to_optimizers.first)) {
      if (tensor_attrs.at(tensor_to_optimizers.first).create_grad ==
          CreateGrad::NO) {
        return false;
      }

      ArrayShape tensor_guid_array_shape =
          array_shape_from_tensor_shape(tensor_attrs.at(tensor_to_optimizers.first).shape);
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

bool are_allocated_tensors_valid(
    AllocatedTensors const &allocated_tensors,
    std::unordered_map<tensor_guid_t, TensorAttrs> const &tensor_attrs) {
  return are_allocated_forward_tensors_valid(allocated_tensors, tensor_attrs) &&
         are_allocated_gradient_tensors_valid(allocated_tensors,
                                              tensor_attrs) &&
         are_allocated_optimizer_tensors_valid(allocated_tensors, tensor_attrs);
}

AllocatedTensors make_empty_allocated_tensors() {
  return AllocatedTensors{{}, {}, {}};
}

} // namespace FlexFlow
