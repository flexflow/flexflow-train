#include "task-spec/task_binding.h"
#include "pcg/tensor_guid_t.dtg.h"
#include "utils/containers/contains_key.h"
#include "utils/fmt/unordered_map.h"
#include "utils/hash/tuple.h"
#include "utils/hash/unordered_map.h"

namespace FlexFlow {

TaskBinding::TaskBinding() : tensor_bindings(), arg_bindings() {}

TaskBinding::TaskBinding(
    std::unordered_map<training_tensor_slot_id_t, symbolic_training_tensor_guid_t> const
        &tensor_bindings,
    std::unordered_map<slot_id_t, TaskArgSpec> const &arg_bindings)
    : tensor_bindings(tensor_bindings), arg_bindings(arg_bindings) {}

void TaskBinding::bind(int name, symbolic_forward_tensor_guid_t const &binding) {
  this->bind(slot_id_t{name}, binding);
}

void TaskBinding::bind(slot_id_t name, symbolic_forward_tensor_guid_t const &binding) {
  this->tensor_bindings.insert({training_tensor_slot_id_t{name, TensorType::FORWARD},
                                symbolic_training_tensor_guid_t{binding}});
}

void TaskBinding::bind_grad(int name, symbolic_gradient_tensor_guid_t const &binding) {
  this->bind_grad(slot_id_t{name}, binding);
}

void TaskBinding::bind_grad(slot_id_t name,
                            symbolic_gradient_tensor_guid_t const &binding) {
  this->tensor_bindings.insert(
      {training_tensor_slot_id_t{name, TensorType::GRADIENT},
       symbolic_training_tensor_guid_t{binding}});
}

void TaskBinding::bind_optimizer(int name,
                                 symbolic_optimizer_tensor_guid_t const &binding) {
  this->bind_optimizer(slot_id_t{name}, binding);
}

void TaskBinding::bind_optimizer(slot_id_t name,
                                 symbolic_optimizer_tensor_guid_t const &binding) {
  this->tensor_bindings.insert(
      {training_tensor_slot_id_t{name, TensorType::OPTIMIZER},
       symbolic_training_tensor_guid_t{binding}});
}

void TaskBinding::bind_loss(int name, symbolic_loss_tensor_guid_t const &binding) {
  this->bind_loss(slot_id_t{name}, binding);
}

void TaskBinding::bind_loss(slot_id_t name, symbolic_loss_tensor_guid_t const &binding) {
  this->tensor_bindings.insert({training_tensor_slot_id_t{name, TensorType::LOSS},
                                symbolic_training_tensor_guid_t{binding}});
}

void TaskBinding::insert_arg_spec(slot_id_t name, TaskArgSpec const &arg_spec) {
  assert(!contains_key(this->arg_bindings, name));
  this->arg_bindings.insert({name, arg_spec});
}

bool TaskBinding::operator==(TaskBinding const &other) const {
  return this->tie() == other.tie();
}

bool TaskBinding::operator!=(TaskBinding const &other) const {
  return this->tie() != other.tie();
}

std::tuple<
    std::unordered_map<training_tensor_slot_id_t, symbolic_training_tensor_guid_t> const &,
    std::unordered_map<slot_id_t, TaskArgSpec> const &>
    TaskBinding::tie() const {
  return std::tie(this->tensor_bindings, this->arg_bindings);
}

std::unordered_map<training_tensor_slot_id_t, symbolic_training_tensor_guid_t> const &
    TaskBinding::get_tensor_bindings() const {
  return this->tensor_bindings;
}

std::unordered_map<slot_id_t, TaskArgSpec> const &
    TaskBinding::get_arg_bindings() const {
  return this->arg_bindings;
}

std::string format_as(TaskBinding const &x) {
  return fmt::format(
    "<TaskBinding tensor_bindings={} arg_bindings={}>",
    x.get_tensor_bindings(),
    x.get_arg_bindings());
}

std::ostream &operator<<(std::ostream &s, TaskBinding const &x) {
  return (s << fmt::to_string(x));
}

} // namespace FlexFlow

namespace std {

size_t hash<::FlexFlow::TaskBinding>::operator()(
    ::FlexFlow::TaskBinding const &s) const {
  return ::FlexFlow::get_std_hash(s.tie());
}

} // namespace std
