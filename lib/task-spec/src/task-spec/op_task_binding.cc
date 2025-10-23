#include "task-spec/op_task_binding.h"
#include "task-spec/op_arg_spec.h"
#include "utils/containers/contains_key.h"

namespace FlexFlow {

void OpTaskBinding::bind(
    int slot, VariadicTensorRef<OpTensorSpec> const &variadic_tensor_ref) {
  this->bind(slot_id_t{slot}, variadic_tensor_ref);
}

void OpTaskBinding::bind(
    slot_id_t slot,
    VariadicTensorRef<OpTensorSpec> const &variadic_tensor_ref) {
  NOT_IMPLEMENTED();
}

void OpTaskBinding::bind(int slot, OpTensorSpec const &tensor_spec) {
  this->bind(slot_id_t{slot}, tensor_spec);
}

void OpTaskBinding::bind(slot_id_t slot, OpTensorSpec const &tensor_spec) {
  this->tensor_bindings.insert({fwb_tensor_slot_id_t{slot, IsGrad::NO}, tensor_spec});
}

void OpTaskBinding::bind_grad(int slot, OpTensorSpec const &tensor_spec) {
  this->bind_grad(slot_id_t{slot}, tensor_spec);
}

void OpTaskBinding::bind_grad(slot_id_t slot, OpTensorSpec const &tensor_spec) {
  this->tensor_bindings.insert({fwb_tensor_slot_id_t{slot, IsGrad::YES}, tensor_spec});
}

void OpTaskBinding::insert_arg_spec(slot_id_t name, OpArgSpec const &arg_spec) {
  assert(!contains_key(this->arg_bindings, name));
  this->arg_bindings.insert({name, arg_spec});
}

bool OpTaskBinding::operator==(OpTaskBinding const &other) const {
  return this->tie() == other.tie();
}

bool OpTaskBinding::operator!=(OpTaskBinding const &other) const {
  return this->tie() != other.tie();
}

std::tuple<std::unordered_map<fwb_tensor_slot_id_t, OpTensorSpec> const &,
           std::unordered_map<slot_id_t, OpArgSpec> const &>
    OpTaskBinding::tie() const {
  return std::tie(this->tensor_bindings, this->arg_bindings);
}

std::unordered_map<fwb_tensor_slot_id_t, OpTensorSpec> const &
    OpTaskBinding::get_tensor_bindings() const {
  return this->tensor_bindings;
}

std::unordered_map<slot_id_t, OpArgSpec> const &
    OpTaskBinding::get_arg_bindings() const {
  return this->arg_bindings;
}

void OpTaskBinding::bind_from_forward(OpTaskBinding const &fwd) {
  this->arg_bindings = fwd.get_arg_bindings();
  this->tensor_bindings = fwd.get_tensor_bindings();
}

OpTaskBinding infer_bwd_binding(OpTaskBinding const &fwd) {
  OpTaskBinding bwd;
  bwd.bind_from_forward(fwd);
  for (auto const &[key, spec] : fwd.get_tensor_bindings()) {
    OpSlotOptions slot_option = spec.slot_option;
    if (slot_option != OpSlotOptions::UNTRAINABLE ||
        slot_option != OpSlotOptions::OPTIONAL_UNTRAINABLE) {
      slot_id_t slot = key.slot_id;
      bwd.bind_grad(slot, spec);
    }
  }
  return bwd;
}

bool is_tensor_invocation_valid(OpTaskSignature const &sig,
                                OpTaskInvocation const &inv) {
  // TODO: fix for variadic inputs (need to implement .bind() for variadic
  // first)
  for (std::pair<fwb_tensor_slot_id_t, OpTensorSpec> const &tensor_binding :
       inv.binding.get_tensor_bindings()) {
    OpTensorSlotSpec op_tensor_slot_spec =
        OpTensorSlotSpec{tensor_binding.first.slot_id,
                         SlotType::TENSOR,
                         tensor_binding.second.role,
                         tensor_binding.first.is_grad,
                         tensor_binding.second.slot_option};

    if (!sig.get_tensor_slots().count(op_tensor_slot_spec)) {
      return false;
    }
  }

  return true;
}

bool is_arg_invocation_valid(OpTaskSignature const &sig,
                             OpTaskInvocation const &inv) {
  // TODO: fix for device specific args
  // for (std::pair<slot_id_t, OpArgSpec> const & arg_binding :
  // inv.binding.get_arg_bindings()) {
  //   if (sig.get_arg_types().count(arg_binding.first)) {
  //     if (get_op_arg_spec_type_index(arg_binding.second) !=
  //     sig.get_arg_types().at(arg_binding.first)) {
  //       return false;
  //     }
  //   } else {
  //     return false;
  //   }
  // }

  return true;
}

bool is_invocation_valid(OpTaskSignature const &sig,
                         OpTaskInvocation const &inv) {
  return is_tensor_invocation_valid(sig, inv) &&
         is_arg_invocation_valid(sig, inv);
}


} // namespace FlexFlow
