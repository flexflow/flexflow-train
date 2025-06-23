#ifndef _FLEXFLOW_LOCAL_EXECUTION_TASK_BINDING_H
#define _FLEXFLOW_LOCAL_EXECUTION_TASK_BINDING_H

#include "task-spec/loss_tensor_guid_t.dtg.h"
#include "task-spec/optimizer_tensor_guid_t.dtg.h"
#include "task-spec/slot_id_t.dtg.h"
#include "task-spec/task_arg_spec.dtg.h"
#include "task-spec/task_id_t.dtg.h"
#include "task-spec/task_signature.dtg.h"
#include "task-spec/tensor_sub_slot_id_t.dtg.h"
#include "task-spec/training_tensor_guid_t.dtg.h"

namespace FlexFlow {

struct TaskBinding {
  TaskBinding();

  explicit TaskBinding(
      std::unordered_map<tensor_sub_slot_id_t, training_tensor_guid_t> const
          &tensor_bindings,
      std::unordered_map<slot_id_t, TaskArgSpec> const &arg_bindings);

  void bind(int, forward_tensor_guid_t const &);
  void bind(slot_id_t, forward_tensor_guid_t const &);

  void bind_grad(int, gradient_tensor_guid_t const &);
  void bind_grad(slot_id_t, gradient_tensor_guid_t const &);

  void bind_optimizer(int, optimizer_tensor_guid_t const &);
  void bind_optimizer(slot_id_t, optimizer_tensor_guid_t const &);

  void bind_loss(int, loss_tensor_guid_t const &);
  void bind_loss(slot_id_t, loss_tensor_guid_t const &);

  template <typename T>
  void bind_arg(int name, T const &t) {
    this->bind_arg<T>(slot_id_t{name}, t);
  }

  template <typename T>
  void bind_arg(slot_id_t name, T const &t) {
    this->insert_arg_spec(name, TaskArgSpec{ConcreteArgSpec::create(t)});
  }

  template <typename T>
  void bind_arg(int name, RuntimeArgRef<T> const &t) {
    this->bind_arg<T>(slot_id_t{name}, t);
  }

  template <typename T>
  void bind_arg(slot_id_t name, RuntimeArgRef<T> const &ref) {
    this->insert_arg_spec(name, TaskArgSpec{RuntimeArgRefSpec::create(ref)});
  }

  bool operator==(TaskBinding const &other) const;
  bool operator!=(TaskBinding const &other) const;

  std::unordered_map<tensor_sub_slot_id_t, training_tensor_guid_t> const &
      get_tensor_bindings() const;
  std::unordered_map<slot_id_t, TaskArgSpec> const &get_arg_bindings() const;
  void insert_arg_spec(slot_id_t name, TaskArgSpec const &arg_spec);

private:
  std::unordered_map<tensor_sub_slot_id_t, training_tensor_guid_t>
      tensor_bindings;
  std::unordered_map<slot_id_t, TaskArgSpec> arg_bindings;

private:
  std::tuple<decltype(tensor_bindings) const &, decltype(arg_bindings) const &>
      tie() const;

  friend ::std::hash<TaskBinding>;
};

std::string format_as(TaskBinding const &x);
std::ostream &operator<<(std::ostream &s, TaskBinding const &x);

} // namespace FlexFlow

namespace std {

template <>
struct hash<::FlexFlow::TaskBinding> {
  size_t operator()(::FlexFlow::TaskBinding const &s) const;
};

} // namespace std

#endif
