#include "local-execution/local_task_argument_accessor.h"
#include "pcg/device_id_t.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/transform.h"
#include "utils/hash/pair.h"
#include "utils/overload.h"

namespace FlexFlow {

LocalTaskArgumentAccessor::LocalTaskArgumentAccessor(
    Allocator const &allocator,
      std::unordered_map<TaskTensorParameter, TensorSlotBacking> const
          &tensor_slots_backing,
      ProfilingSettings const &profiling_settings,
      device_handle_t const &ff_handle,
      DeviceType kernel_device_type,
      PCGOperatorAttrs const &op_attrs,
      std::optional<LossAttrs> const &loss_attrs,
      std::optional<PerDeviceOpState> const &per_device_op_state,
      FFIterationConfig const &iteration_config,
      std::optional<OptimizerAttrs> const &optimizer_attrs,
      size_t device_idx)
    : allocator(allocator), 
      tensor_slots_backing(tensor_slots_backing),
      profiling_settings(profiling_settings),
      ff_handle(ff_handle),
      kernel_device_type(kernel_device_type),
      op_attrs(op_attrs),
      loss_attrs(loss_attrs),
      per_device_op_state(per_device_op_state),
      iteration_config(iteration_config),
      optimizer_attrs(optimizer_attrs),
      device_idx(make_device_id_t_from_idx(nonnegative_int{device_idx}, kernel_device_type))
{ }

GenericTensorAccessor LocalTaskArgumentAccessor::get_tensor(
    TensorSlotName slot, Permissions priv, TrainingTensorType tensor_type) const {
  GenericTensorAccessorW tensor_backing =
      this->tensor_slots_backing.at(slot_tensor_type).require_single();
  if (priv == Permissions::RO) {
    GenericTensorAccessorR readonly_tensor_backing =
        read_only_accessor_from_write_accessor(tensor_backing);
    return readonly_tensor_backing;
  } else if (priv == Permissions::RW || priv == Permissions::WO) {
    return tensor_backing;
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
  return this->kernel_device_type;
}

PCGOperatorAttrs LocalTaskArgumentAccessor::get_op_attrs() const {
  return this->op_attrs;
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

size_t LocalTaskArgumentAccessor::get_device_idx() const {
  return this->device_idx;
}

} // namespace FlexFlow
