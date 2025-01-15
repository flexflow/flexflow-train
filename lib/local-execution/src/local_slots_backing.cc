#include "local-execution/local_slots_backing.h"
#include "local-execution/tensor_reduction.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "pcg/computation_graph.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/map_values.h"
#include "utils/overload.h"

namespace FlexFlow {

LocalSlotsBacking::LocalSlotsBacking(
    LayerTensorBackingMap const &allocated_forward_tensors,
    TensorBackingMap const &allocated_non_graph_tensors,
    RuntimeArgConfig const &runtime_arg_config)
    : tensor_mapping(allocated_forward_tensors),
      non_graph_tensor_mapping(allocated_non_graph_tensors),
      runtime_arg_config(runtime_arg_config){};

void LocalSlotsBacking::add_per_device_op_state(
    layer_guid_t const &op_guid,
    DeviceSpecificDeviceStates const &device_state) {
  this->per_device_op_states.insert({op_guid, device_state});
}

void LocalSlotsBacking::allocate_layer_tensors(
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

void LocalSlotsBacking::allocate_tensors_by_role(
    TensorRole const &role,
    layer_guid_t const &layer_guid,
    ComputationGraph const &computation_graph,
    Allocator &allocator) {
  std::vector<tensor_guid_t> tensors;
  switch (role) {
    case TensorRole::INPUT:
      tensors = get_incoming_inputs(computation_graph, layer_guid);
      this->input_tensor_slots.insert({layer_guid, 
        transform(tensors, [&](tensor_guid_t const &tensor_guid) {
          return lower(tensor_guid);
        })
      });
      break;
    case TensorRole::WEIGHT:
      tensors = get_incoming_weights(computation_graph, layer_guid);
      this->weight_tensor_slots.insert({layer_guid, 
        transform(tensors, [&](tensor_guid_t const &tensor_guid) {
          return lower(tensor_guid);
        })
      });
      break;
    case TensorRole::OUTPUT:
      tensors = get_outgoing_tensors(computation_graph, layer_guid);
      this->output_tensor_slots.insert({layer_guid, 
        transform(tensors, [&](tensor_guid_t const &tensor_guid) {
          return lower(tensor_guid);
        })
      });
      break;
    default:
      throw mk_runtime_error("Invalid tensor role, got {}", role);
  }

  for (tensor_guid_t const &tensor : tensors) {
    TensorAttrs tensor_attrs = get_tensor_attrs(computation_graph, tensor);
    reduced_tensor_t reduced_tensor = lower(tensor);
    LayerTensorKey layer_tensor_key =
        LayerTensorKey{layer_guid, reduced_tensor};
    // tensor allocation
    if (!is_forward_tensor_allocated(layer_tensor_key)) {
      GenericTensorAccessorW tensor_backing =
          allocator.allocate_tensor(tensor_attrs.shape);
      this->tensor_mapping.insert({layer_tensor_key, tensor_backing});
    }

    // gradient tensor allocation
    if (tensor_attrs.create_gradients == CreateGrad::YES) {
      GenericTensorAccessorW gradient_tensor_backing =
          allocator.allocate_tensor(tensor_attrs.shape);
      this->gradient_tensor_mapping.insert(
          {layer_tensor_key, gradient_tensor_backing});
    }
  }
}

void LocalSlotsBacking::allocate_optimizer_tensors(
    layer_guid_t const &weight_layer,
    tensor_guid_t const &weight,
    ComputationGraph const &cg,
    Allocator &allocator,
    TaskSignature const &sig) {
  GenericTensorAccessorW weight_backing = this->get_tensor_backing(
      TensorType::FORWARD, lower(weight), weight_layer);
  int num_grad_buffer_tensors =
      sig.tensor_guid_slots.size() - 2; // ignore 2 (weight and weight_grad)
  std::vector<reduced_tensor_t> optimizer_buffer_tensors;
  for (int i = 0; i < num_grad_buffer_tensors; ++i) {
    reduced_tensor_t buffer_tensor = reduced_tensor_t{i};
    GenericTensorAccessorW buffer_backing = allocator.allocate_tensor(
        get_tensor_shape(weight_backing.shape, weight_backing.data_type));
    this->optimizer_tensor_mapping.insert(
        {LayerTensorKey{weight_layer, buffer_tensor}, buffer_backing});
    optimizer_buffer_tensors.push_back(buffer_tensor);
  }
  this->weight_optimizer_tensor_guids.insert(
      {weight_layer, optimizer_buffer_tensors});
}

bool LocalSlotsBacking::is_forward_tensor_allocated(
    LayerTensorKey const &layer_tensor_id) const {
  return contains_key(this->tensor_mapping, layer_tensor_id);
}

bool LocalSlotsBacking::is_non_graph_tensor_allocated(
    reduced_tensor_t const &tensor_id) const {
  return contains_key(this->non_graph_tensor_mapping, tensor_id);
}

GenericTensorAccessorW const &LocalSlotsBacking::get_tensor_backing(
    TensorType const &tensor_type,
    reduced_tensor_t const &tensor_id,
    std::optional<layer_guid_t> const &layer_guid) const {
  switch (tensor_type) {
    case TensorType::FORWARD:
      return this->tensor_mapping.at(
          LayerTensorKey{layer_guid.value(), tensor_id});
    case TensorType::NON_GRAPH:
      return this->non_graph_tensor_mapping.at(tensor_id);
    case TensorType::GRADIENT:
      return this->gradient_tensor_mapping.at(
          LayerTensorKey{layer_guid.value(), tensor_id});
    case TensorType::OPTIMIZER:
      return this->optimizer_tensor_mapping.at(
          LayerTensorKey{layer_guid.value(), tensor_id});
    default:
      throw mk_runtime_error(
          fmt::format("Invalid tensor type {}", tensor_type));
  }
}

TensorSlotsBacking LocalSlotsBacking::construct_tensor_slots_backing(
    OpTaskBinding const &binding, layer_guid_t const &op_guid) const {
  TensorSlotsBacking mapping;

  for (auto const &tensor_binding : binding.get_tensor_bindings()) {
    SlotTensorTypeId slot_grad_id = tensor_binding.first;
    OpTensorSpec tensor_spec = tensor_binding.second;
    std::vector<reduced_tensor_t> tensor_guids;
    int weight_adjusted_idx = 0;
    switch (tensor_spec.role) {
      case TensorRole::WEIGHT:
        assert(contains_key(this->weight_tensor_slots, op_guid));
        tensor_guids = this->weight_tensor_slots.at(op_guid);
        break;
      case TensorRole::INPUT:
        assert(contains_key(this->input_tensor_slots, op_guid));
        tensor_guids = this->input_tensor_slots.at(op_guid);
        break;
      case TensorRole::OUTPUT:
        assert(contains_key(this->output_tensor_slots, op_guid));
        tensor_guids = this->output_tensor_slots.at(op_guid);
        break;
      default:
        throw mk_runtime_error(
            fmt::format("Invalid TensorRole {}", tensor_spec.role));
    }

    mapping.insert({slot_grad_id,
                    this->get_tensor_backing(slot_grad_id.tensor_type,
                                             tensor_guids.at(tensor_spec.idx),
                                             op_guid)});
  }
  return mapping;
}

TensorSlotsBacking LocalSlotsBacking::construct_tensor_slots_backing(
    TaskBinding const &binding,
    std::optional<layer_guid_t> const &layer_guid) const {
  TensorSlotsBacking mapping;

  for (auto const &tensor_binding : binding.get_tensor_bindings()) {
    reduced_tensor_t tensor_id = tensor_binding.second;
    SlotTensorTypeId slot_tensor_type_id = tensor_binding.first;
    GenericTensorAccessorW accessor = this->get_tensor_backing(
        slot_tensor_type_id.tensor_type, tensor_id, layer_guid);
    mapping.insert({slot_tensor_type_id, accessor});
  }

  return mapping;
}

ArgSlotsBacking LocalSlotsBacking::construct_arg_slots_backing(
    OpTaskBinding const &binding, layer_guid_t const &op_guid) const {
  return map_values(
      binding.get_arg_bindings(), [&](OpArgSpec const &arg_binding) {
        return arg_binding.template visit<ConcreteArgSpec>(
            overload{[&](OpArgRefSpec const &s) {
                       return this->resolve_op_arg_ref_spec(s, op_guid);
                     },
                     [&](RuntimeArgRefSpec const &s) {
                       return this->resolve_runtime_arg_ref_spec(s);
                     },
                     [](ConcreteArgSpec const &s) { return s; }});
      });
}

ArgSlotsBacking LocalSlotsBacking::construct_arg_slots_backing(
    TaskBinding const &binding) const {
  return map_values(
      binding.get_arg_bindings(), [&](TaskArgSpec const &arg_binding) {
        return arg_binding.template visit<ConcreteArgSpec>(
            overload{[&](RuntimeArgRefSpec const &s) {
                       return this->resolve_runtime_arg_ref_spec(s);
                     },
                     [](ConcreteArgSpec const &s) { return s; }});
      });
  ;
}

ConcreteArgSpec LocalSlotsBacking::resolve_op_arg_ref_spec(
    OpArgRefSpec const &op_arg_ref_spec, layer_guid_t const &op_guid) const {
  if (op_arg_ref_spec.holds<DeviceSpecificDeviceStates>()) {
    assert(contains_key(per_device_op_states, op_guid));
    DeviceSpecificDeviceStates device_specific =
        per_device_op_states.at(op_guid);
    PerDeviceOpState device_state =
        get_device_state_from_device_specific(device_specific, 0);
    return ConcreteArgSpec::create(device_state);
  } else if (op_arg_ref_spec.holds<ParallelTensorShape>()) {
    ParallelTensorShapeRefType index_op_arg_ref =
        op_arg_ref_spec.get_ref_type().get<ParallelTensorShapeRefType>();

    assert(contains_key(this->input_tensor_slots, op_guid));
    std::vector<reduced_tensor_t> input_tensor_guids =
        this->input_tensor_slots.at(op_guid);

    assert(input_tensor_guids.size() > index_op_arg_ref.idx);
    GenericTensorAccessorW tensor_backing =
        this->get_tensor_backing(TensorType::FORWARD,
                                 input_tensor_guids.at(index_op_arg_ref.idx),
                                 op_guid);
    ParallelTensorShape shape = lift_to_parallel(
        get_tensor_shape(tensor_backing.shape, tensor_backing.data_type));
    return ConcreteArgSpec::create(shape);
  } else {
    throw mk_runtime_error("Unhandled op arg ref type");
  }
}

ConcreteArgSpec LocalSlotsBacking::resolve_runtime_arg_ref_spec(
    RuntimeArgRefSpec const &runtime_arg_ref_spec) const {
  if (runtime_arg_ref_spec.holds<DeviceSpecific<PerDeviceFFHandle>>()) {
    return ConcreteArgSpec::create(
        *(this->runtime_arg_config.ff_handle.get(0)));
  } else if (runtime_arg_ref_spec.holds<ProfilingSettings>()) {
    return ConcreteArgSpec::create(this->runtime_arg_config.profiling_settings);
  } else {
    throw mk_runtime_error("Unhandled runtime arg ref type");
  }
}

} // namespace FlexFlow
