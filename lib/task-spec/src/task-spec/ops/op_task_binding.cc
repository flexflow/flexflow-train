#include "task-spec/ops/op_task_binding.h"
#include "task-spec/ops/op_arg_spec.h"
#include "utils/containers/contains_key.h"

namespace FlexFlow {

void OpTaskBinding::bind(
    TensorSlotName slot,
    VariadicTensorRef<OpTensorSpec> const &variadic_tensor_ref) {
  NOT_IMPLEMENTED();
}

void OpTaskBinding::bind(TensorSlotName slot, OpTensorSpec const &tensor_spec) {
  this->tensor_bindings.insert({slot, tensor_spec});
}

void OpTaskBinding::insert_arg_spec(arg_slot_id_t name, OpArgSpec const &arg_spec) {
  assert(!contains_key(this->arg_bindings, name));
  this->arg_bindings.insert({name, arg_spec});
}

bool OpTaskBinding::operator==(OpTaskBinding const &other) const {
  return this->tie() == other.tie();
}

bool OpTaskBinding::operator!=(OpTaskBinding const &other) const {
  return this->tie() != other.tie();
}

std::tuple<std::unordered_map<TensorSlotName, OpTensorSpec> const &,
           std::unordered_map<arg_slot_id_t, OpArgSpec> const &>
    OpTaskBinding::tie() const {
  return std::tie(this->tensor_bindings, this->arg_bindings);
}

std::unordered_map<TensorSlotName, OpTensorSpec> const &
    OpTaskBinding::get_tensor_bindings() const {
  return this->tensor_bindings;
}

std::unordered_map<arg_slot_id_t, OpArgSpec> const &
    OpTaskBinding::get_arg_bindings() const {
  return this->arg_bindings;
}

void OpTaskBinding::bind_from_forward(OpTaskBinding const &fwd) {
  this->arg_bindings = fwd.get_arg_bindings();
  this->tensor_bindings = fwd.get_tensor_bindings();
}

OpTaskBinding infer_bwd_binding(OpTaskBinding const &fwd) {
  // TODO(@lockshaw)(#pr): 
  NOT_IMPLEMENTED();
  // OpTaskBinding bwd;
  // bwd.bind_from_forward(fwd);
  // for (auto const &[key, spec] : fwd.get_tensor_bindings()) {
  //   OpSlotOptions slot_option = spec.slot_option;
  //   if (slot_option != OpSlotOptions::UNTRAINABLE ||
  //       slot_option != OpSlotOptions::OPTIONAL_UNTRAINABLE) {
  //     slot_id_t slot = key.slot_id;
  //     bwd.bind_grad(slot, spec);
  //   }
  // }
  // return bwd;
}

} // namespace FlexFlow
