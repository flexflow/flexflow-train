#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_TASK_ARGUMENT_ACCESSOR_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_TASK_ARGUMENT_ACCESSOR_H

#include "task-spec/device_specific.h"
#include "task-spec/itask_argument_accessor.h"
#include "task-spec/per_device_op_state.dtg.h"

namespace FlexFlow {

struct TaskArgumentAccessor {
  // arguments
  template <typename T>
  T const &get_argument(slot_id_t slot) const {
    return this->ptr->get_concrete_arg(slot).get<T>();
  }

  template <typename T>
  T const &get_argument(int slot) const {
    return this->get_argument<T>(slot_id_t{slot});
  }

  // tensors
  template <Permissions PRIV>
  privilege_mode_to_accessor<PRIV> get_tensor(int slot) const {
    return this->get_tensor<PRIV>(slot_id_t{slot});
  }

  template <Permissions PRIV>
  privilege_mode_to_accessor<PRIV> get_tensor(slot_id_t slot) const {
    return std::get<privilege_mode_to_accessor<PRIV>>(
        this->ptr->get_tensor(slot, PRIV, TensorType::FORWARD));
  }

  template <Permissions PRIV>
  privilege_mode_to_accessor<PRIV> get_tensor_grad(int slot) const {
    return this->get_tensor_grad<PRIV>(slot_id_t{slot});
  }

  template <Permissions PRIV>
  privilege_mode_to_accessor<PRIV> get_tensor_grad(slot_id_t slot) const {
    return std::get<privilege_mode_to_accessor<PRIV>>(
        this->ptr->get_tensor(slot, PRIV, TensorType::GRADIENT));
  }

  template <Permissions PRIV>
  privilege_mode_to_accessor<PRIV> get_optimizer_tensor(int slot) const {
    return this->get_optimizer_tensor<PRIV>(slot_id_t{slot});
  }

  template <Permissions PRIV>
  privilege_mode_to_accessor<PRIV> get_optimizer_tensor(slot_id_t slot) const {
    return std::get<privilege_mode_to_accessor<PRIV>>(
        this->ptr->get_tensor(slot, PRIV, TensorType::OPTIMIZER));
  }

  template <Permissions PRIV>
  privilege_mode_to_accessor<PRIV> get_loss_tensor(int slot) const {
    return this->get_loss_tensor<PRIV>(slot_id_t{slot});
  }

  template <Permissions PRIV>
  privilege_mode_to_accessor<PRIV> get_loss_tensor(slot_id_t slot) const {
    return std::get<privilege_mode_to_accessor<PRIV>>(
        this->ptr->get_tensor(slot, PRIV, TensorType::LOSS));
  }

  // variadic tensors
  template <Permissions PRIV>
  std::vector<privilege_mode_to_accessor<PRIV>>
      get_variadic_tensor(int slot) const {
    return this->get_variadic_tensor<PRIV>(slot_id_t{slot});
  }

  template <Permissions PRIV>
  std::vector<privilege_mode_to_accessor<PRIV>>
      get_variadic_tensor(slot_id_t slot) const {
    return std::get<std::vector<privilege_mode_to_accessor<PRIV>>>(
        this->ptr->get_variadic_tensor(slot, PRIV, TensorType::FORWARD));
  }

  template <Permissions PRIV>
  std::vector<privilege_mode_to_accessor<PRIV>>
      get_variadic_tensor_grad(int slot) const {
    return this->get_variadic_tensor_grad<PRIV>(slot_id_t{slot});
  }

  template <Permissions PRIV>
  std::vector<privilege_mode_to_accessor<PRIV>>
      get_variadic_tensor_grad(slot_id_t slot) const {
    return std::get<std::vector<privilege_mode_to_accessor<PRIV>>>(
        this->ptr->get_variadic_tensor(slot, PRIV, TensorType::GRADIENT));
  }

  template <Permissions PRIV>
  std::vector<privilege_mode_to_accessor<PRIV>>
      get_variadic_optimizer_tensor(int slot) const {
    return this->get_variadic_optimizer_tensor<PRIV>(slot_id_t{slot});
  }

  template <Permissions PRIV>
  std::vector<privilege_mode_to_accessor<PRIV>>
      get_variadic_optimizer_tensor(slot_id_t slot) const {
    return std::get<std::vector<privilege_mode_to_accessor<PRIV>>>(
        this->ptr->get_variadic_tensor(slot, PRIV, TensorType::OPTIMIZER));
  }

  template <Permissions PRIV>
  std::vector<privilege_mode_to_accessor<PRIV>>
      get_variadic_loss_tensor(int slot) const {
    return this->get_variadic_loss_tensor<PRIV>(slot_id_t{slot});
  }

  template <Permissions PRIV>
  std::vector<privilege_mode_to_accessor<PRIV>>
      get_variadic_loss_tensor(slot_id_t slot) const {
    return std::get<std::vector<privilege_mode_to_accessor<PRIV>>>(
        this->ptr->get_variadic_tensor(slot, PRIV, TensorType::LOSS));
  }

  Allocator get_allocator() const {
    return this->ptr->get_allocator();
  }

  template <typename T, typename... Args>
  static
      typename std::enable_if<std::is_base_of<ITaskArgumentAccessor, T>::value,
                              TaskArgumentAccessor>::type
      create(Args &&...args) {
    return TaskArgumentAccessor(
        std::make_shared<T>(std::forward<Args>(args)...));
  }

private:
  TaskArgumentAccessor(std::shared_ptr<ITaskArgumentAccessor const> ptr)
      : ptr(ptr) {}
  std::shared_ptr<ITaskArgumentAccessor const> ptr;
};

} // namespace FlexFlow

#endif
