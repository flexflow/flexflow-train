#ifndef _FLEXFLOW_LOCAL_EXECUTION_LOCAL_TASK_ARGUMENT_ACCESSOR_H
#define _FLEXFLOW_LOCAL_EXECUTION_LOCAL_TASK_ARGUMENT_ACCESSOR_H

#include "local-execution/tensor_slot_backing.dtg.h"
#include "pcg/device_id_t.dtg.h"
#include "task-spec/runtime_arg_config.dtg.h"
#include "task-spec/task_argument_accessor.h"
#include "task-spec/training_tensor_slot_id_t.dtg.h"
#include <unordered_map>
#include <variant>

namespace FlexFlow {

struct LocalTaskArgumentAccessor : public ITaskArgumentAccessor {
  explicit LocalTaskArgumentAccessor(
      Allocator const &allocator,
      std::unordered_map<training_tensor_slot_id_t, TensorSlotBacking> const
          &tensor_slots_backing,
      std::unordered_map<slot_id_t, ConcreteArgSpec> const &arg_slots_backing,
      size_t device_idx);

  LocalTaskArgumentAccessor(LocalTaskArgumentAccessor const &) = delete;
  LocalTaskArgumentAccessor(LocalTaskArgumentAccessor &&) = delete;

  ConcreteArgSpec const &get_concrete_arg(slot_id_t) const override;

  GenericTensorAccessor get_tensor(slot_id_t slot,
                                   Permissions priv,
                                   TrainingTensorType tensor_type) const override;
  VariadicGenericTensorAccessor get_variadic_tensor(
      slot_id_t slot, Permissions priv, TrainingTensorType tensor_type) const override;

  Allocator get_allocator() const override;

  size_t get_device_idx() const override;

private:
  Allocator allocator;
  std::unordered_map<training_tensor_slot_id_t, TensorSlotBacking>
      tensor_slots_backing;
  std::unordered_map<slot_id_t, ConcreteArgSpec> arg_slots_backing;
  size_t device_idx; 
};

CHECK_RC_COPY_VIRTUAL_COMPLIANT(LocalTaskArgumentAccessor);

} // namespace FlexFlow

#endif
