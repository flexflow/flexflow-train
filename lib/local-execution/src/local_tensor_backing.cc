#include "local-execution/local_tensor_backing.h"
#include "local-execution/tensor_lowering.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "pcg/computation_graph.h"
#include "utils/containers/contains_key.h"
#include "utils/overload.h"
#include "local-execution/slot_grad_id.dtg.h"

namespace FlexFlow {

LocalTensorBacking::LocalTensorBacking() {};

void LocalTensorBacking::allocate_layer_tensors(
    layer_guid_t const &layer_guid,
    ComputationGraph const &computation_graph,
    Allocator &allocator) {
  this->allocate_tensors_by_role(
      TensorRole::INPUT, layer_guid, computation_graph, allocator);
  this->allocate_tensors_by_role(
      TensorRole::WEIGHT, layer_guid, computation_graph, allocator);
  this->allocate_tensors_by_role(
      TensorRole::OUTPUT, layer_guid, computation_graph, allocator);
}

void LocalTensorBacking::allocate_tensors_by_role(
    TensorRole const &role,
    layer_guid_t const &layer_guid,
    ComputationGraph const &computation_graph,
    Allocator &allocator) {
  std::vector<tensor_guid_t> tensors;
  switch (role) {
    case TensorRole::INPUT:
      tensors = get_incoming_inputs(computation_graph, layer_guid);
      break;
    case TensorRole::WEIGHT:
      tensors = get_incoming_weights(computation_graph, layer_guid);
      break;
    case TensorRole::OUTPUT:
      tensors = get_outgoing_tensors(computation_graph, layer_guid);
      break;
    default:
      throw mk_runtime_error("Invalid tensor role, got {}", role);
  }

  for (tensor_guid_t const &tensor : tensors) {
    TensorAttrs tensor_attrs = get_tensor_attrs(computation_graph, tensor);
    // tensor allocation
    if (!contains_key(this->tensor_lowering_mapping, tensor)) {
      lowered_tensor_t reduced_tensor = this->lowered_tensor_source.new_lowered_tensor();
      this->tensor_lowering_mapping.insert({tensor, reduced_tensor});
      GenericTensorAccessorW tensor_backing =
          allocator.allocate_tensor(tensor_attrs.shape);
      this->tensor_backings.insert({reduced_tensor, tensor_backing});
    }

    // gradient tensor allocation
    if (tensor_attrs.create_gradients == CreateGrad::YES && !contains_key(this->gradient_tensor_lowering_mapping, tensor)) {
      lowered_tensor_t reduced_tensor = this->lowered_tensor_source.new_lowered_tensor();
      this->gradient_tensor_lowering_mapping.insert({tensor, reduced_tensor});
      GenericTensorAccessorW gradient_tensor_backing =
          allocator.allocate_tensor(tensor_attrs.shape);
      this->tensor_backings.insert(
          {reduced_tensor, gradient_tensor_backing});
    }
  }
}

void LocalTensorBacking::allocate_optimizer_tensors(
    tensor_guid_t const &weight,
    std::vector<optimizer_tensor_t> const& optimizer_tensors,
    Allocator &allocator) {
  GenericTensorAccessorW weight_backing = this->get_tensor_backing(this->tensor_lowering_mapping.at(weight));
  for (optimizer_tensor_t const & optimizer_tensor: optimizer_tensors) {
    // optimizer tensor allocation
    if (!contains_key(this->optimizer_tensor_lowering_mapping, optimizer_tensor)) {
      lowered_tensor_t buffer_tensor = this->lowered_tensor_source.new_lowered_tensor();
      this->optimizer_tensor_lowering_mapping.insert({optimizer_tensor, buffer_tensor});
      GenericTensorAccessorW buffer_backing = allocator.allocate_tensor(
          get_tensor_shape(weight_backing.shape, weight_backing.data_type));
      this->tensor_backings.insert({buffer_tensor, buffer_backing});
    }
  }
}

bool LocalTensorBacking::is_tensor_allocated(lowered_tensor_t const & tensor_id) const {
  return contains_key(tensor_backings, tensor_id);
}

GenericTensorAccessorW const &LocalTensorBacking::get_tensor_backing(
    lowered_tensor_t const &tensor_id) const {
  return this->tensor_backings.at(tensor_id);
}

TensorSlotsBacking LocalTensorBacking::construct_tensor_slots_backing(
    TaskBinding const &binding) const {
  TensorSlotsBacking mapping;

  for (auto const &tensor_binding : binding.get_tensor_bindings()) {
    SlotTensorTypeId slot_tensor_type_id = tensor_binding.first;

    lowered_tensor_t tensor_id = [&] {
      TensorTypeVariant tensor_type = tensor_binding.second;
      if (tensor_type.has<tensor_guid_t>() and slot_tensor_type_id.tensor_type == TensorType::FORWARD) {
        return this->tensor_lowering_mapping.at(tensor_type.get<tensor_guid_t>());
      } else if (tensor_type.has<tensor_guid_t>() and slot_tensor_type_id.tensor_type == TensorType::GRADIENT) {
        return this->gradient_tensor_lowering_mapping.at(tensor_type.get<tensor_guid_t>());
      } else if (tensor_type.has<optimizer_tensor_t>()) {
        return this->optimizer_tensor_lowering_mapping.at(tensor_type.get<optimizer_tensor_t>());
      } else if (tensor_type.has<loss_tensor_t>()) {
        return this->loss_tensor_lowering_mapping.at(tensor_type.get<loss_tensor_t>());
      } else {
        throw mk_runtime_error(fmt::format("Tensor binding has invalid type"));
      }
    }();

    GenericTensorAccessorW accessor = this->get_tensor_backing(tensor_id);
    mapping.insert({slot_tensor_type_id, accessor});
  }

  return mapping;
}

} // namespace FlexFlow
