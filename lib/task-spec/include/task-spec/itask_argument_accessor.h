#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_ITASK_ARGUMENT_ACCESSOR_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_ITASK_ARGUMENT_ACCESSOR_H

#include "kernels/allocation.h"
#include "task-spec/privilege_tensor_accessor.h"
#include "task-spec/concrete_arg.h"
#include "task-spec/op_task_signature.h"
#include "task-spec/tensor_type.dtg.h"

namespace FlexFlow {

struct ITaskArgumentAccessor {
  ITaskArgumentAccessor &operator=(ITaskArgumentAccessor const &) = delete;

  virtual ~ITaskArgumentAccessor() = default;

  virtual ConcreteArgSpec const &get_concrete_arg(slot_id_t) const = 0;

  virtual GenericTensorAccessor get_tensor(slot_id_t slot,
                                           Permissions priv,
                                           TensorType tensor_type) const = 0;
  virtual VariadicGenericTensorAccessor get_variadic_tensor(
      slot_id_t slot, Permissions priv, TensorType tensor_type) const = 0;

  virtual Allocator get_allocator() const = 0;
  virtual size_t get_device_idx() const = 0;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(ITaskArgumentAccessor);

} // namespace FlexFlow

#endif
