#ifndef _FLEXFLOW_REALM_BACKEND_REALM_TASK_ARGUMENT_ACCESSOR_H
#define _FLEXFLOW_REALM_BACKEND_REALM_TASK_ARGUMENT_ACCESSOR_H

#include "realm-backend/realm_allocator.h"
#include "task-spec/slot_tensor_type_id.dtg.h"
#include "task-spec/task_argument_accessor.h"
#include <unordered_map>
#include <variant>

namespace FlexFlow {

using TensorSlotsBacking = std::unordered_map<
    SlotTensorTypeId,
    std::variant<GenericTensorAccessorW, std::vector<GenericTensorAccessorW>>>;
using ArgSlotsBacking = std::unordered_map<slot_id_t, ConcreteArgSpec>;

struct RealmTaskArgumentAccessor : public ITaskArgumentAccessor {
  RealmTaskArgumentAccessor(Allocator const &allocator,
                            TensorSlotsBacking const &tensor_slots_backing,
                            ArgSlotsBacking const &arg_slots_backing);

  RealmTaskArgumentAccessor(RealmTaskArgumentAccessor const &) = delete;
  RealmTaskArgumentAccessor(RealmTaskArgumentAccessor &&) = delete;

  ConcreteArgSpec const &get_concrete_arg(slot_id_t) const override;

  GenericTensorAccessor get_tensor(slot_id_t slot, Permissions priv,
                                   TensorType tensor_type) const override;
  VariadicGenericTensorAccessor
  get_variadic_tensor(slot_id_t slot, Permissions priv,
                      TensorType tensor_type) const override;

  Allocator get_allocator() const override;

  size_t get_device_idx() const override;

private:
  Allocator allocator;
  TensorSlotsBacking tensor_slots_backing;
  ArgSlotsBacking arg_slots_backing;
};

CHECK_RC_COPY_VIRTUAL_COMPLIANT(RealmTaskArgumentAccessor);

} // namespace FlexFlow

#endif
