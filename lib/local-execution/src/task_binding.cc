#include "task-spec/task_binding.h"
#include "pcg/tensor_guid_t.dtg.h"
#include "task-spec/training_tensor_guid_t.dtg.h"
#include "utils/containers/contains_key.h"
#include "utils/fmt/unordered_map.h"
#include "utils/hash/tuple.h"
#include "utils/hash/unordered_map.h"

namespace FlexFlow {

TaskBinding::TaskBinding() : tensor_bindings(), arg_bindings() {}

TaskBinding::TaskBinding(
    std::unordered_map<tensor_sub_slot_id_t, training_tensor_guid_t> const
        &tensor_bindings,
    std::unordered_map<slot_id_t, TaskArgSpec> const &arg_bindings)
    : tensor_bindings(tensor_bindings), arg_bindings(arg_bindings) {}

void TaskBinding::bind(int name, forward_tensor_guid_t const &binding) {
  this->bind(slot_id_t{name}, binding);
}

void TaskBinding::bind(slot_id_t name, forward_tensor_guid_t const &binding) {
  this->tensor_bindings.insert({tensor_sub_slot_id_t{name, TensorType::FORWARD},
                                training_tensor_guid_t{binding}});
}

void TaskBinding::bind_grad(int name, gradient_tensor_guid_t const &binding) {
  this->bind_grad(slot_id_t{name}, binding);
}

void TaskBinding::bind_grad(slot_id_t name,
                            gradient_tensor_guid_t const &binding) {
  this->tensor_bindings.insert(
      {tensor_sub_slot_id_t{name, TensorType::GRADIENT},
       training_tensor_guid_t{binding}});
}

void TaskBinding::bind_optimizer(int name,
                                 optimizer_tensor_guid_t const &binding) {
  this->bind_optimizer(slot_id_t{name}, binding);
}

void TaskBinding::bind_optimizer(slot_id_t name,
                                 optimizer_tensor_guid_t const &binding) {
  this->tensor_bindings.insert(
      {tensor_sub_slot_id_t{name, TensorType::OPTIMIZER},
       training_tensor_guid_t{binding}});
}

void TaskBinding::bind_loss(int name, loss_tensor_guid_t const &binding) {
  this->bind_loss(slot_id_t{name}, binding);
}

void TaskBinding::bind_loss(slot_id_t name, loss_tensor_guid_t const &binding) {
  this->tensor_bindings.insert({tensor_sub_slot_id_t{name, TensorType::LOSS},
                                training_tensor_guid_t{binding}});
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
    std::unordered_map<tensor_sub_slot_id_t, training_tensor_guid_t> const &,
    std::unordered_map<slot_id_t, TaskArgSpec> const &>
    TaskBinding::tie() const {
  return std::tie(this->tensor_bindings, this->arg_bindings);
}

std::unordered_map<tensor_sub_slot_id_t, training_tensor_guid_t> const &
    TaskBinding::get_tensor_bindings() const {
  return this->tensor_bindings;
}

std::unordered_map<slot_id_t, TaskArgSpec> const &
    TaskBinding::get_arg_bindings() const {
  return this->arg_bindings;
}

std::string format_as(TaskBinding const &x) {
  std::ostringstream oss;
  oss << "<TaskBinding";
  oss << " tensor_bindings=" << x.get_tensor_bindings();
  oss << " arg_bindings=" << x.get_arg_bindings();
  return oss.str();
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
