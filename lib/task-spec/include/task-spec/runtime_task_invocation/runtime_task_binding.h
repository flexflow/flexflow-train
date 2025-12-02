#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_RUNTIME_TASK_INVOCATION_RUNTIME_TASK_BINDING_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_RUNTIME_TASK_INVOCATION_RUNTIME_TASK_BINDING_H

#include "task-spec/dynamic_graph/dynamic_tensor_slot.dtg.h"
#include "task-spec/symbolic/symbolic_loss_tensor_guid_t.dtg.h"
#include "task-spec/symbolic/symbolic_optimizer_tensor_guid_t.dtg.h"
#include "task-spec/symbolic/symbolic_forward_tensor_guid_t.dtg.h"
#include "task-spec/symbolic/symbolic_gradient_tensor_guid_t.dtg.h"
#include "task-spec/ops/arg_slot_id_t.dtg.h"
#include "task-spec/runtime_task_invocation/runtime_arg_spec.dtg.h"
#include "task-spec/task_id_t.dtg.h"
#include "task-spec/symbolic/symbolic_training_tensor_guid_t.dtg.h"
#include "task-spec/task_argument_accessor/task_tensor_parameter.dtg.h"
#include "task-spec/runtime_task_invocation/runtime_arg_ref.h"
#include "task-spec/optimizer_slot_name.dtg.h"

namespace FlexFlow {

struct RuntimeTaskBinding {
  RuntimeTaskBinding();

  explicit RuntimeTaskBinding(
      std::unordered_map<DynamicTensorSlot, symbolic_training_tensor_guid_t> const
          &tensor_bindings,
      std::unordered_map<arg_slot_id_t, RuntimeArgSpec> const &arg_bindings);

  void bind(TensorSlotName, symbolic_forward_tensor_guid_t const &);
  void bind_grad(TensorSlotName, symbolic_gradient_tensor_guid_t const &);
  void bind_optimizer(TensorSlotName, OptimizerSlotName, symbolic_optimizer_tensor_guid_t const &);
  void bind_loss(symbolic_loss_tensor_guid_t const &);

  template <typename T>
  void bind_arg(int name, T const &t) {
    this->bind_arg<T>(arg_slot_id_t{name}, t);
  }

  template <typename T>
  void bind_arg(arg_slot_id_t name, T const &t) {
    this->insert_arg_spec(name, RuntimeArgSpec{ConcreteArgSpec::create(t)});
  }

  template <typename T>
  void bind_arg(int name, RuntimeArgRef<T> const &t) {
    this->bind_arg<T>(arg_slot_id_t{name}, t);
  }

  template <typename T>
  void bind_arg(arg_slot_id_t name, RuntimeArgRef<T> const &ref) {
    this->insert_arg_spec(name, RuntimeArgSpec{RuntimeArgRefSpec::create(ref)});
  }

  bool operator==(RuntimeTaskBinding const &other) const;
  bool operator!=(RuntimeTaskBinding const &other) const;

  // std::unordered_map<FwbTensorSlot, > const &
  //     get_tensor_bindings() const;
  std::unordered_map<arg_slot_id_t, RuntimeArgSpec> const &get_arg_bindings() const;
  void insert_arg_spec(arg_slot_id_t name, RuntimeArgSpec const &arg_spec);

private:
  // std::unordered_map<FwbTensorSlot, symbolic_training_tensor_guid_t>
  //     tensor_bindings;
  std::unordered_map<arg_slot_id_t, RuntimeArgSpec> arg_bindings;

private:
  // std::tuple<decltype(tensor_bindings) const &, decltype(arg_bindings) const &>
  //     tie() const;

  friend ::std::hash<RuntimeTaskBinding>;
};

std::string format_as(RuntimeTaskBinding const &x);
std::ostream &operator<<(std::ostream &s, RuntimeTaskBinding const &x);

} // namespace FlexFlow

namespace std {

template <>
struct hash<::FlexFlow::RuntimeTaskBinding> {
  size_t operator()(::FlexFlow::RuntimeTaskBinding const &s) const;
};

} // namespace std

#endif
