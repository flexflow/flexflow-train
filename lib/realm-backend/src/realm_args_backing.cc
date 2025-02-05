#include "op-attrs/parallel_tensor_shape.h"
#include "realm-backend/realm_args_backing.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/map_values.h"
#include "utils/overload.h"

namespace FlexFlow {

void RealmArgsBacking::add_per_device_op_state(
    layer_guid_t const &op_guid, Future<DeviceSpecificDeviceStates> &&future) {
  if (per_device_op_states.find(op_guid) != per_device_op_states.end()) {
    throw mk_runtime_error("Op state already exists");
  }
  per_device_op_states.insert({op_guid, std::move(future)});
}

ArgSlotsBacking RealmArgsBacking::construct_arg_slots_backing(
    TaskBinding const &binding) const {
  return map_values(binding.get_arg_bindings(),
                    [&](TaskArgSpec const &arg_binding) {
                      return arg_binding.template visit<ConcreteArgSpec>(
                          overload{[&](RuntimeArgRefSpec const &s) {
                                     return this->lower_to_concrete_arg_spec(s);
                                   },
                                   [](ConcreteArgSpec const &s) { return s; }});
                    });
  ;
}

ConcreteArgSpec RealmArgsBacking::lower_to_concrete_arg_spec(
    OpArgRefSpec const &op_arg_ref_spec, ComputationGraph const &cg,
    layer_guid_t const &op_guid) const {
  if (op_arg_ref_spec.holds<DeviceSpecificDeviceStates>()) {
    assert(contains_key(this->per_device_op_states, op_guid));
    DeviceSpecificDeviceStates device_specific =
        per_device_op_states.at(op_guid);
    PerDeviceOpState device_state =
        get_device_state_from_device_specific(device_specific, 0);
    return ConcreteArgSpec::create(device_state);
  } else if (op_arg_ref_spec.holds<ParallelTensorShape>()) {
    ParallelTensorShapeRefType index_op_arg_ref =
        op_arg_ref_spec.get_ref_type().get<ParallelTensorShapeRefType>();
    tensor_guid_t input_tensor =
        get_incoming_inputs(cg, op_guid).at(index_op_arg_ref.idx);
    TensorAttrs tensor_attrs = get_tensor_attrs(cg, input_tensor);
    ParallelTensorShape shape = lift_to_parallel(tensor_attrs.shape);
    return ConcreteArgSpec::create(shape);
  } else {
    throw mk_runtime_error("Unhandled op arg ref type");
  }
}

ConcreteArgSpec RealmArgsBacking::lower_to_concrete_arg_spec(
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
