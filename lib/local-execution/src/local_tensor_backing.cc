#include "local-execution/local_tensor_backing.h"
#include "task-spec/slot_grad_id.dtg.h"

#include "local-execution/allocated_tensors.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "pcg/computation_graph.h"
#include "pcg/optimizer_attrs.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/keys.h"
#include "utils/overload.h"

namespace FlexFlow {

LocalTensorBacking::LocalTensorBacking(
    AllocatedTensors const &allocated_tensors,
    UnallocatedTensors const &unallocated_tensors,
    Allocator const &allocator)
    : tensor_gradient_mapping(allocated_tensors.gradient_mapping),
      tensor_optimizer_mapping(allocated_tensors.optimizer_mapping),
      allocator(allocator) {

  // handle already-allocated tensors
  for (std::pair<TensorTypeVariant, GenericTensorAccessorW> const
           &tensor_type_backing : allocated_tensors.tensor_type_backings) {
    lowered_tensor_t lowered_tensor =
        this->insert_tensor(tensor_type_backing.first);
    this->tensor_backings.insert({lowered_tensor, tensor_type_backing.second});
  }

  // allocate new tensors
  this->tensor_gradient_mapping.insert(
      unallocated_tensors.gradient_mapping.begin(),
      unallocated_tensors.gradient_mapping.end());

  for (std::pair<tensor_guid_t, std::vector<optimizer_tensor_t>> const
           &unallocated_optimizer_tensors :
       unallocated_tensors.optimizer_mapping) {
    if (this->tensor_optimizer_mapping.count(
            unallocated_optimizer_tensors.first)) {
      for (optimizer_tensor_t const &optimizer_tensor :
           unallocated_optimizer_tensors.second) {
        this->tensor_optimizer_mapping[unallocated_optimizer_tensors.first]
            .push_back(optimizer_tensor);
      }
    } else {
      this->tensor_optimizer_mapping.insert({unallocated_optimizer_tensors});
    }
  }

  for (std::pair<TensorTypeVariant, TensorShape> const &tensor_type_shape :
       unallocated_tensors.tensor_type_shapes) {
    lowered_tensor_t lowered_tensor =
        this->insert_tensor(tensor_type_shape.first);
    GenericTensorAccessorW tensor_backing =
        this->allocator.allocate_tensor(tensor_type_shape.second);
    this->tensor_backings.insert({lowered_tensor, tensor_backing});
  }
};

lowered_tensor_t
    LocalTensorBacking::insert_tensor(TensorTypeVariant const &tensor_type) {
  lowered_tensor_t lowered_tensor =
      this->lowered_tensor_source.new_lowered_tensor();
  tensor_type.visit<std::nullopt_t>(overload{
      [&](tensor_guid_t const &tensor_guid) {
        this->tensor_lowering_mapping.insert({tensor_guid, lowered_tensor});
        return std::nullopt;
      },
      [&](gradient_tensor_t const &gradient_tensor) {
        this->gradient_tensor_lowering_mapping.insert(
            {gradient_tensor, lowered_tensor});
        return std::nullopt;
      },
      [&](optimizer_tensor_t const &optimizer_tensor) {
        this->optimizer_tensor_lowering_mapping.insert(
            {optimizer_tensor, lowered_tensor});
        return std::nullopt;
      },
      [&](loss_tensor_t const &loss_tensor) {
        this->loss_tensor_lowering_mapping.insert(
            {loss_tensor, lowered_tensor});
        return std::nullopt;
      },
      [&](auto const &any_tensor) {
        throw mk_runtime_error(
            fmt::format("Unhandled tensor type {}", any_tensor));
      }});
  return lowered_tensor;
}

GenericTensorAccessorW
    LocalTensorBacking::get_tensor(TensorTypeVariant const &tensor_type) const {
  lowered_tensor_t lowered_tensor =
      tensor_type.visit<lowered_tensor_t>(overload{
          [&](tensor_guid_t const &tensor_guid) {
            return this->tensor_lowering_mapping.at(tensor_guid);
          },
          [&](gradient_tensor_t const &gradient_tensor) {
            return this->gradient_tensor_lowering_mapping.at(gradient_tensor);
          },
          [&](optimizer_tensor_t const &optimizer_tensor) {
            return this->optimizer_tensor_lowering_mapping.at(optimizer_tensor);
          },
          [&](loss_tensor_t const &loss_tensor) {
            return this->loss_tensor_lowering_mapping.at(loss_tensor);
          },
          [&](auto const &any_tensor) {
            throw mk_runtime_error(
                fmt::format("Unhandled tensor type {}", any_tensor));
          }});
  return this->tensor_backings.at(lowered_tensor);
}

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

    if (tensor_attrs.create_gradients == CreateGrad::YES &&
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
    if (tensor_attrs.create_gradients == CreateGrad::YES) {
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

TensorSlotsBacking construct_tensor_slots_backing(
    LocalTensorBacking const &local_tensor_backing,
    TaskBinding const &binding) {
  TensorSlotsBacking mapping;

  for (std::pair<SlotTensorTypeId, TensorTypeVariant> const &tensor_binding :
       binding.get_tensor_bindings()) {
    mapping.insert({tensor_binding.first,
                    local_tensor_backing.get_tensor(tensor_binding.second)});
  }

  return mapping;
}

} // namespace FlexFlow
