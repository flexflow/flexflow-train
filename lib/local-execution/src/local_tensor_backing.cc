#include "local-execution/local_tensor_backing.h"
#include "local-execution/slot_grad_id.dtg.h"
#include "local-execution/tensor_lowering.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "pcg/computation_graph.h"
#include "pcg/optimizer_attrs.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/keys.h"
#include "utils/overload.h"

namespace FlexFlow {

LocalTensorBacking::LocalTensorBacking(
    std::unordered_map<TensorTypeVariant, GenericTensorAccessorW> const
        &allocated_tensor_backings,
    std::unordered_set<tensor_guid_t> const &allocated_tensor_guids,
    std::unordered_map<tensor_guid_t, gradient_tensor_t> const
        &allocated_gradient_mapping,
    std::unordered_map<tensor_guid_t, std::vector<optimizer_tensor_t>> const
        &allocated_optimizer_mapping,
    std::unordered_set<loss_tensor_t> const &allocated_loss_tensors)
    : tensor_gradient_mapping(allocated_gradient_mapping),
      tensor_optimizer_mapping(allocated_optimizer_mapping) {

  // computation graph tensors
  for (tensor_guid_t const &allocated_tensor_guid : allocated_tensor_guids) {
    lowered_tensor_t lowered_tensor = this->insert_tensor(
        allocated_tensor_backings.at(TensorTypeVariant{allocated_tensor_guid}));
    this->tensor_lowering_mapping.insert(
        {allocated_tensor_guid, lowered_tensor});
  }

  // gradient tensors
  for (std::pair<tensor_guid_t, gradient_tensor_t> const
           &tensor_guid_gradient_pair : allocated_gradient_mapping) {
    gradient_tensor_t allocated_gradient_tensor =
        tensor_guid_gradient_pair.second;
    lowered_tensor_t lowered_tensor =
        this->insert_tensor(allocated_tensor_backings.at(
            TensorTypeVariant{allocated_gradient_tensor}));
    this->gradient_tensor_lowering_mapping.insert(
        {allocated_gradient_tensor, lowered_tensor});
  }

  // optimizer tensors
  for (std::pair<tensor_guid_t, std::vector<optimizer_tensor_t>> const
           &tensor_guid_optimizers_pair : allocated_optimizer_mapping) {
    for (optimizer_tensor_t const &allocated_optimizer_tensor :
         tensor_guid_optimizers_pair.second) {
      lowered_tensor_t lowered_tensor =
          this->insert_tensor(allocated_tensor_backings.at(
              TensorTypeVariant{allocated_optimizer_tensor}));
      this->optimizer_tensor_lowering_mapping.insert(
          {allocated_optimizer_tensor, lowered_tensor});
    }
  }

  // loss tensors
  for (loss_tensor_t const &allocated_loss_tensor : allocated_loss_tensors) {
    lowered_tensor_t lowered_tensor = this->insert_tensor(
        allocated_tensor_backings.at(TensorTypeVariant{allocated_loss_tensor}));
    this->loss_tensor_lowering_mapping.insert(
        {allocated_loss_tensor, lowered_tensor});
  }

  // sanity check that backings match up with the mappings
  assert(this->tensor_backings.size() == allocated_tensor_backings.size());
};

lowered_tensor_t LocalTensorBacking::insert_tensor(
    GenericTensorAccessorW const &tensor_backing) {
  lowered_tensor_t lowered_tensor =
      this->lowered_tensor_source.new_lowered_tensor();
  this->tensor_backings.insert({lowered_tensor, tensor_backing});
  return lowered_tensor;
}

lowered_tensor_t
    LocalTensorBacking::allocate_tensor(TensorShape const &tensor_shape,
                                        Allocator &allocator) {
  GenericTensorAccessorW tensor_backing =
      allocator.allocate_tensor(tensor_shape);
  return this->insert_tensor(tensor_backing);
}

void allocate_tensor_guid(LocalTensorBacking &local_tensor_backing,
                          tensor_guid_t const &tensor_guid,
                          TensorShape const &tensor_shape,
                          Allocator &allocator) {
  if (!contains_key(local_tensor_backing.tensor_lowering_mapping,
                    tensor_guid)) {
    lowered_tensor_t lowered_tensor =
        local_tensor_backing.allocate_tensor(tensor_shape, allocator);
    local_tensor_backing.tensor_lowering_mapping.insert(
        {tensor_guid, lowered_tensor});
  }
}

void allocate_gradient_tensor(LocalTensorBacking &local_tensor_backing,
                              gradient_tensor_t const &gradient_tensor,
                              tensor_guid_t const &tensor_guid,
                              TensorShape const &tensor_shape,
                              Allocator &allocator) {
  if (!contains_key(local_tensor_backing.tensor_gradient_mapping,
                    tensor_guid)) {
    local_tensor_backing.tensor_gradient_mapping.insert(
        {tensor_guid, gradient_tensor});
    lowered_tensor_t lowered_tensor =
        local_tensor_backing.allocate_tensor(tensor_shape, allocator);
    local_tensor_backing.gradient_tensor_lowering_mapping.insert(
        {gradient_tensor, lowered_tensor});
  }
}

void allocate_optimizer_tensors(
    LocalTensorBacking &local_tensor_backing,
    std::vector<optimizer_tensor_t> const &optimizer_tensors,
    tensor_guid_t const &tensor_guid,
    TensorShape const &tensor_shape,
    Allocator &allocator) {
  if (!contains_key(local_tensor_backing.tensor_optimizer_mapping,
                    tensor_guid)) {
    // insert new optimizer tensors into mappings
    std::vector<optimizer_tensor_t> optimizer_tensors;
    for (optimizer_tensor_t const &optimizer_tensor : optimizer_tensors) {
      // allocate lowered tensor
      lowered_tensor_t lowered_tensor =
          local_tensor_backing.allocate_tensor(tensor_shape, allocator);
      local_tensor_backing.optimizer_tensor_lowering_mapping.insert(
          {optimizer_tensor, lowered_tensor});
    }
    local_tensor_backing.tensor_optimizer_mapping.insert(
        {tensor_guid, optimizer_tensors});
  }
}

void allocate_loss_tensor(LocalTensorBacking &local_tensor_backing,
                          loss_tensor_t const &loss_tensor,
                          TensorShape const &tensor_shape,
                          Allocator &allocator) {
  lowered_tensor_t lowered_tensor =
      local_tensor_backing.allocate_tensor(tensor_shape, allocator);
  local_tensor_backing.loss_tensor_lowering_mapping.insert(
      {loss_tensor, lowered_tensor});
}

void allocate_all_computation_graph_tensors(
    LocalTensorBacking &local_tensor_backing,
    GradientTensorSource &gradient_tensor_source,
    ComputationGraph const &computation_graph,
    Allocator &allocator) {
  // allocate each layer's tensors and gradient tensors
  for (tensor_guid_t const &tensor_guid : get_all_tensors(computation_graph)) {
    TensorAttrs tensor_attrs = get_tensor_attrs(computation_graph, tensor_guid);
    allocate_tensor_guid(
        local_tensor_backing, tensor_guid, tensor_attrs.shape, allocator);

    if (tensor_attrs.create_gradients == CreateGrad::YES) {
      gradient_tensor_t gradient_tensor =
          gradient_tensor_source.new_gradient_tensor();
      allocate_gradient_tensor(local_tensor_backing,
                               gradient_tensor,
                               tensor_guid,
                               tensor_attrs.shape,
                               allocator);
    }
  }
}

void allocate_all_optimizer_tensors(
    LocalTensorBacking &local_tensor_backing,
    OptimizerTensorSource &optimizer_tensor_source,
    ComputationGraph const &computation_graph,
    Allocator &allocator,
    OptimizerAttrs const &optimizer_attrs) {
  for (tensor_guid_t const &tensor_guid : get_all_tensors(computation_graph)) {
    TensorAttrs tensor_attrs = get_tensor_attrs(computation_graph, tensor_guid);
    if (tensor_attrs.create_gradients == CreateGrad::YES) {
      std::vector<optimizer_tensor_t> optimizer_tensors;
      for (int i = 0; i < get_num_optimizer_tensors(optimizer_attrs); ++i) {
        optimizer_tensors.push_back(
            optimizer_tensor_source.new_optimizer_tensor());
      }
      allocate_optimizer_tensors(local_tensor_backing,
                                 optimizer_tensors,
                                 tensor_guid,
                                 tensor_attrs.shape,
                                 allocator);
    }
  }
}

loss_tensor_t allocate_loss_tensor(LocalTensorBacking &local_tensor_backing,
                                   LossTensorSource &loss_tensor_source,
                                   TensorShape const &tensor_shape,
                                   Allocator &allocator) {
  loss_tensor_t loss_tensor = loss_tensor_source.new_loss_tensor();
  lowered_tensor_t lowered_tensor =
      local_tensor_backing.allocate_tensor(tensor_shape, allocator);
  local_tensor_backing.loss_tensor_lowering_mapping.insert(
      {loss_tensor, lowered_tensor});
  return loss_tensor;
}

TensorSlotsBacking construct_tensor_slots_backing(
    LocalTensorBacking const &local_tensor_backing,
    TaskBinding const &binding) {
  TensorSlotsBacking mapping;

  for (auto const &tensor_binding : binding.get_tensor_bindings()) {
    SlotTensorTypeId slot_tensor_type_id = tensor_binding.first;

    lowered_tensor_t lowered_tensor =
        tensor_binding.second.visit<lowered_tensor_t>(overload{
            [&](tensor_guid_t const &t) {
              return local_tensor_backing.tensor_lowering_mapping.at(t);
            },
            [&](gradient_tensor_t const &t) {
              return local_tensor_backing.gradient_tensor_lowering_mapping.at(
                  t);
            },
            [&](optimizer_tensor_t const &t) {
              return local_tensor_backing.optimizer_tensor_lowering_mapping.at(
                  t);
            },
            [&](loss_tensor_t const &t) {
              return local_tensor_backing.loss_tensor_lowering_mapping.at(t);
            },
        });

    GenericTensorAccessorW accessor =
        local_tensor_backing.tensor_backings.at(lowered_tensor);
    mapping.insert({slot_tensor_type_id, accessor});
  }

  return mapping;
}

} // namespace FlexFlow
