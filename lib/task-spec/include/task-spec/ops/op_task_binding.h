#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_OP_TASK_BINDING_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_OP_TASK_BINDING_H

#include "task-spec/ops/op_arg_ref.h"
#include "task-spec/ops/op_arg_spec.dtg.h"
#include "task-spec/ops/op_tensor_spec.h"
#include "op-attrs/tensor_slot_name.dtg.h"
#include "task-spec/ops/arg_slot_id_t.dtg.h"
#include "task-spec/variadic_tensor_ref.h"
#include "task-spec/runtime_task_invocation/runtime_arg_ref.h"

namespace FlexFlow {

struct OpTaskBinding {
  OpTaskBinding() = default;

  void bind(TensorSlotName, VariadicTensorRef<OpTensorSpec> const &);
  void bind(TensorSlotName, OpTensorSpec const &);
  void bind_grad(TensorSlotName, OpTensorSpec const &);

  template <typename T>
  void bind_device_specific_arg(int name, T const &t) {
    this->bind_device_specific_arg<T>(arg_slot_id_t{name}, t);
  }

  template <typename T>
  void bind_device_specific_arg(arg_slot_id_t name, T const &t) {
    this->bind_device_specific_arg<T>(name, t);
  }

  template <typename T>
  void bind_device_specific_arg(int name, OpArgRef<T> const &t) {
    this->bind_device_specific_arg<T>(arg_slot_id_t{name}, t);
  }

  template <typename T>
  void bind_device_specific_arg(arg_slot_id_t name, OpArgRef<T> const &t) {
    this->bind_device_specific_arg<T>(name, t);
  }

  template <typename T>
  void bind_concrete_arg(int name, T const &t) {
    this->bind_arg(arg_slot_id_t{name}, t);
  }

  template <typename T>
  void bind_concrete_arg(arg_slot_id_t name, T const &t) {
    this->insert_arg_spec(name, OpArgSpec{ConcreteArgSpec::create(t)});
  }

  template <typename T>
  void bind_arg(int name, RuntimeArgRef<T> const &ref) {
    this->bind_arg(arg_slot_id_t{name}, ref);
  }

  template <typename T>
  void bind_arg(arg_slot_id_t name, RuntimeArgRef<T> const &ref) {
    this->insert_arg_spec(name, OpArgSpec{RuntimeArgRefSpec::create(ref)});
  }

  template <typename T>
  void bind_arg(int name, OpArgRef<T> const &ref) {
    this->bind_arg(arg_slot_id_t{name}, ref);
  }

  template <typename T>
  void bind_arg(arg_slot_id_t name, OpArgRef<T> const &ref) {
    this->insert_arg_spec(name, OpArgSpec{OpArgRefSpec::create(ref)});
  }

  bool operator==(OpTaskBinding const &other) const;
  bool operator!=(OpTaskBinding const &other) const;

  std::unordered_map<TensorSlotName, OpTensorSpec> const &
      get_tensor_bindings() const;
  std::unordered_map<arg_slot_id_t, OpArgSpec> const &get_arg_bindings() const;

  void bind_from_forward(OpTaskBinding const &fwd);

private:
  std::unordered_map<TensorSlotName, OpTensorSpec> tensor_bindings;
  std::unordered_map<arg_slot_id_t, OpArgSpec> arg_bindings;

private:
  void insert_arg_spec(arg_slot_id_t name, OpArgSpec const &arg_spec);
  std::tuple<decltype(tensor_bindings) const &, decltype(arg_bindings) const &>
      tie() const;
};

OpTaskBinding infer_bwd_binding(OpTaskBinding const &);

} // namespace FlexFlow

#endif
