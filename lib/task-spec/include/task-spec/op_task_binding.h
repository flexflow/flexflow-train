#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OP_TASK_BINDING_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OP_TASK_BINDING_H

#include "task-spec/op_arg_ref.h"
#include "task-spec/op_arg_spec.dtg.h"
#include "task-spec/op_tensor_spec.h"
#include "task-spec/slot_grad_id.dtg.h"
#include "task-spec/slot_id_t.dtg.h"
#include "task-spec/variadic_tensor_ref.h"

namespace FlexFlow {

struct OpTaskBinding {
  OpTaskBinding() = default;

  void bind(int, VariadicTensorRef<OpTensorSpec> const &);
  void bind(slot_id_t, VariadicTensorRef<OpTensorSpec> const &);

  void bind(int, OpTensorSpec const &);
  void bind(slot_id_t, OpTensorSpec const &);

  void bind_grad(int, OpTensorSpec const &);
  void bind_grad(slot_id_t, OpTensorSpec const &);

  template <typename T>
  void bind_device_specific_arg(int name, T const &t) {
    this->bind_device_specific_arg<T>(slot_id_t{name}, t);
  }

  template <typename T>
  void bind_device_specific_arg(slot_id_t name, T const &t) {
    NOT_IMPLEMENTED();
  }

  template <typename T>
  void bind_device_specific_arg(int name, OpArgRef<T> const &t) {
    this->bind_device_specific_arg<T>(slot_id_t{name}, t);
  }

  template <typename T>
  void bind_device_specific_arg(slot_id_t name, OpArgRef<T> const &t) {
    NOT_IMPLEMENTED();
  }

  template <typename T>
  void bind_arg(int name, T const &t) {
    this->bind_arg<T>(slot_id_t{name}, t);
  }

  template <typename T>
  void bind_arg(slot_id_t name, T const &t) {
    this->insert_arg_spec(name, OpArgSpec{ConcreteArgSpec::create(t)});
  }

  template <typename T>
  void bind_arg(int name, RuntimeArgRef<T> const &t) {
    this->bind_arg<T>(slot_id_t{name}, t);
  }

  template <typename T>
  void bind_arg(slot_id_t name, RuntimeArgRef<T> const &ref) {
    this->insert_arg_spec(name, OpArgSpec{RuntimeArgRefSpec::create(ref)});
  }

  template <typename T>
  void bind_arg(int name, OpArgRef<T> const &t) {
    this->bind_arg<T>(slot_id_t{name}, t);
  }

  template <typename T>
  void bind_arg(slot_id_t name, OpArgRef<T> const &ref) {
    this->insert_arg_spec(name, OpArgSpec{OpArgRefSpec::create(ref)});
  }
  bool operator==(OpTaskBinding const &other) const;
  bool operator!=(OpTaskBinding const &other) const;

  std::unordered_map<SlotGradId, OpTensorSpec> const &
      get_tensor_bindings() const;
  std::unordered_map<slot_id_t, OpArgSpec> const &get_arg_bindings() const;

  void bind_from_forward(OpTaskBinding const &fwd);

private:
  std::unordered_map<SlotGradId, OpTensorSpec> tensor_bindings;
  std::unordered_map<slot_id_t, OpArgSpec> arg_bindings;

private:
  void insert_arg_spec(slot_id_t name, OpArgSpec const &arg_spec);
  std::tuple<decltype(tensor_bindings) const &, decltype(arg_bindings) const &>
      tie() const;
};

OpTaskBinding infer_bwd_binding(OpTaskBinding const &fwd);

} // namespace FlexFlow

#endif
