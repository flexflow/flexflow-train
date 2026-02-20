#include "local-execution/local_task_argument_accessor.h"
#include "kernels/accessor.h"
#include "pcg/device_id.h"
#include "pcg/device_id_t.h"
#include "utils/exception.h"
#include "utils/optional.h"

namespace FlexFlow {

LocalTaskArgumentAccessor::LocalTaskArgumentAccessor(
    Allocator const &allocator,
    std::unordered_map<TaskTensorParameter, DynamicTensorAccessor> const
        &tensor_slots_backing,
    ProfilingSettings const &profiling_settings,
    device_handle_t const &ff_handle,
    std::optional<PCGOperatorAttrs> const &op_attrs,
    std::optional<LossAttrs> const &loss_attrs,
    std::optional<PerDeviceOpState> const &per_device_op_state,
    FFIterationConfig const &iteration_config,
    std::optional<OptimizerAttrs> const &optimizer_attrs,
    device_id_t device_idx)
    : allocator(allocator), tensor_slots_backing(tensor_slots_backing),
      profiling_settings(profiling_settings), ff_handle(ff_handle),
      op_attrs(op_attrs), loss_attrs(loss_attrs),
      per_device_op_state(per_device_op_state),
      iteration_config(iteration_config), optimizer_attrs(optimizer_attrs),
      device_idx(device_idx) {}

GenericTensorAccessor
    LocalTaskArgumentAccessor::get_tensor(TaskTensorParameter slot,
                                          Permissions priv) const {
  DynamicTensorAccessor tensor_backing = this->tensor_slots_backing.at(slot);
  if (priv == Permissions::RO) {
    if (tensor_backing.is_read()) {
      return tensor_backing.require_read();
    } else {
      GenericTensorAccessorR readonly_tensor_backing =
          read_only_accessor_from_write_accessor(
              tensor_backing.require_write());
      return readonly_tensor_backing;
    }
  } else if (priv == Permissions::RW || priv == Permissions::WO) {
    return tensor_backing.require_write();
  } else {
    PANIC(fmt::format("Unhandled privilege mode {}", priv));
  }
}

ProfilingSettings LocalTaskArgumentAccessor::get_profiling_settings() const {
  return this->profiling_settings;
}

device_handle_t LocalTaskArgumentAccessor::get_ff_handle() const {
  return this->ff_handle;
}

DeviceType LocalTaskArgumentAccessor::get_kernel_device_type() const {
  return get_device_type(this->device_idx);
}

PCGOperatorAttrs LocalTaskArgumentAccessor::get_op_attrs() const {
  return assert_unwrap(this->op_attrs);
}

LossAttrs LocalTaskArgumentAccessor::get_loss_attrs() const {
  return assert_unwrap(this->loss_attrs);
}

PerDeviceOpState LocalTaskArgumentAccessor::get_per_device_op_state() const {
  return assert_unwrap(this->per_device_op_state);
}

FFIterationConfig LocalTaskArgumentAccessor::get_iteration_config() const {
  return this->iteration_config;
}

OptimizerAttrs LocalTaskArgumentAccessor::get_optimizer_attrs() const {
  return assert_unwrap(this->optimizer_attrs);
}

Allocator LocalTaskArgumentAccessor::get_allocator() const {
  return this->allocator;
}

device_id_t LocalTaskArgumentAccessor::get_device_idx() const {
  return this->device_idx;
}

} // namespace FlexFlow
