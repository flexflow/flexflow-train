#include "task-spec/runtime_task_binding.h"
#include "pcg/tensor_guid_t.dtg.h"
#include "utils/containers/contains_key.h"
#include "utils/fmt/unordered_map.h"
#include "utils/hash/tuple.h"
#include "utils/hash/unordered_map.h"

namespace FlexFlow {

RuntimeTaskBinding::RuntimeTaskBinding() : tensor_bindings(), arg_bindings() {}

RuntimeTaskBinding::RuntimeTaskBinding(
    std::unordered_map<training_tensor_slot_id_t, symbolic_training_tensor_guid_t> const
        &tensor_bindings,
    std::unordered_map<slot_id_t, RuntimeArgSpec> const &arg_bindings)
    : tensor_bindings(tensor_bindings), arg_bindings(arg_bindings) {}

void RuntimeTaskBinding::bind(int name, symbolic_forward_tensor_guid_t const &binding) {
  this->bind(slot_id_t{name}, binding);
}

void RuntimeTaskBinding::bind(slot_id_t name, symbolic_forward_tensor_guid_t const &binding) {
  this->tensor_bindings.insert({training_tensor_slot_id_t{name, TrainingTensorType::FORWARD},
                                symbolic_training_tensor_guid_t{binding}});
}

void RuntimeTaskBinding::bind_grad(int name, symbolic_gradient_tensor_guid_t const &binding) {
  this->bind_grad(slot_id_t{name}, binding);
}

void RuntimeTaskBinding::bind_grad(slot_id_t name,
                            symbolic_gradient_tensor_guid_t const &binding) {
  this->tensor_bindings.insert(
      {training_tensor_slot_id_t{name, TrainingTensorType::GRADIENT},
       symbolic_training_tensor_guid_t{binding}});
}

void RuntimeTaskBinding::bind_optimizer(int name,
                                 symbolic_optimizer_tensor_guid_t const &binding) {
  this->bind_optimizer(slot_id_t{name}, binding);
}

void RuntimeTaskBinding::bind_optimizer(slot_id_t name,
                                 symbolic_optimizer_tensor_guid_t const &binding) {
  this->tensor_bindings.insert(
      {training_tensor_slot_id_t{name, TrainingTensorType::OPTIMIZER},
       symbolic_training_tensor_guid_t{binding}});
}

void RuntimeTaskBinding::bind_loss(int name, symbolic_loss_tensor_guid_t const &binding) {
  this->bind_loss(slot_id_t{name}, binding);
}

void RuntimeTaskBinding::bind_loss(slot_id_t name, symbolic_loss_tensor_guid_t const &binding) {
  this->tensor_bindings.insert({training_tensor_slot_id_t{name, TrainingTensorType::LOSS},
                                symbolic_training_tensor_guid_t{binding}});
}

void RuntimeTaskBinding::insert_arg_spec(slot_id_t name, RuntimeArgSpec const &arg_spec) {
  assert(!contains_key(this->arg_bindings, name));
  this->arg_bindings.insert({name, arg_spec});
}

bool RuntimeTaskBinding::operator==(RuntimeTaskBinding const &other) const {
  return this->tie() == other.tie();
}

bool RuntimeTaskBinding::operator!=(RuntimeTaskBinding const &other) const {
  return this->tie() != other.tie();
}

std::tuple<
    std::unordered_map<training_tensor_slot_id_t, symbolic_training_tensor_guid_t> const &,
    std::unordered_map<slot_id_t, RuntimeArgSpec> const &>
    RuntimeTaskBinding::tie() const {
  return std::tie(this->tensor_bindings, this->arg_bindings);
}

std::unordered_map<training_tensor_slot_id_t, symbolic_training_tensor_guid_t> const &
    RuntimeTaskBinding::get_tensor_bindings() const {
  return this->tensor_bindings;
}

std::unordered_map<slot_id_t, RuntimeArgSpec> const &
    RuntimeTaskBinding::get_arg_bindings() const {
  return this->arg_bindings;
}

std::string format_as(RuntimeTaskBinding const &x) {
  return fmt::format(
    "<RuntimeTaskBinding tensor_bindings={} arg_bindings={}>",
    x.get_tensor_bindings(),
    x.get_arg_bindings());
}

std::ostream &operator<<(std::ostream &s, RuntimeTaskBinding const &x) {
  return (s << fmt::to_string(x));
}

} // namespace FlexFlow

namespace std {

size_t hash<::FlexFlow::RuntimeTaskBinding>::operator()(
    ::FlexFlow::RuntimeTaskBinding const &s) const {
  return ::FlexFlow::get_std_hash(s.tie());
}

} // namespace std
