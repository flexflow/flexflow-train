#include "task-spec/ops/impl/element_binary.h"
#include "kernels/element_binary_kernels.h"
#include "task-spec/profiling.h"
#include "utils/hash-utils.h"

namespace FlexFlow {

using namespace FlexFlow::Kernels::ElementBinary;

static DeviceSpecificPerDeviceOpState
    init_task_impl(TaskArgumentAccessor const &acc) {
  auto input_lhs = acc.get_tensor<Permissions::RO>(TensorSlotName::LHS_INPUT);
  auto input_rhs = acc.get_tensor<Permissions::RO>(TensorSlotName::RHS_INPUT);
  auto output = acc.get_tensor<Permissions::WO>(TensorSlotName::OUTPUT);

  device_handle_t handle = acc.get_ff_handle();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  ElementBinaryAttrs attrs = acc.get_op_attrs().require_element_binary();

  std::optional<ElementBinaryPerDeviceState> per_device_state =
      init_kernel(kernel_device_type,
                  handle,
                  attrs.type,
                  attrs.should_broadcast_lhs,
                  attrs.should_broadcast_rhs,
                  input_lhs.shape,
                  input_rhs.shape,
                  output.shape);

  return DeviceSpecificPerDeviceOpState{
    acc.make_device_specific(per_device_state),
  };
}

static std::optional<milliseconds_t> forward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  ElementBinaryPerDeviceState per_device_state = acc.get_per_device_op_state().require_element_binary().value();
  ElementBinaryAttrs attrs = acc.get_op_attrs().require_element_binary();
  device_handle_t handle = acc.get_ff_handle();

  auto input_lhs = acc.get_tensor<Permissions::RO>(TensorSlotName::LHS_INPUT);
  auto input_rhs = acc.get_tensor<Permissions::RO>(TensorSlotName::RHS_INPUT);
  auto output = acc.get_tensor<Permissions::WO>(TensorSlotName::OUTPUT);

  return profile(forward_kernel,
                 profiling,
                 kernel_device_type,
                 "[ElementBinary] forward_time = {:.2lf}ms\n",
                 per_device_state,
                 input_lhs.get_float_ptr(),
                 input_rhs.get_float_ptr(),
                 output.get_float_ptr(),
                 attrs.type,
                 attrs.should_broadcast_lhs,
                 handle);
}

static std::optional<milliseconds_t>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  ElementBinaryPerDeviceState per_device_state = acc.get_per_device_op_state().require_element_binary().value();
  ElementBinaryAttrs attrs = acc.get_op_attrs().require_element_binary();
  device_handle_t handle = acc.get_ff_handle();

  auto input_lhs = acc.get_tensor<Permissions::RO>(TensorSlotName::LHS_INPUT);
  auto input_rhs = acc.get_tensor<Permissions::RO>(TensorSlotName::RHS_INPUT);

  auto output_grad = acc.get_tensor_grad<Permissions::RO>(TensorSlotName::OUTPUT);
  auto input_lhs_grad = acc.get_tensor_grad<Permissions::RW>(TensorSlotName::LHS_INPUT);
  auto input_rhs_grad = acc.get_tensor_grad<Permissions::RW>(TensorSlotName::RHS_INPUT);

  return profile(backward_kernel,
                 profiling,
                 kernel_device_type,
                 "[ElementBinary] backward_time = {:.2lf}ms\n",
                 per_device_state,
                 output_grad.get_float_ptr(),
                 input_lhs.get_float_ptr(),
                 input_rhs.get_float_ptr(),
                 input_lhs_grad.get_float_ptr(),
                 input_rhs_grad.get_float_ptr(),
                 attrs.type,
                 attrs.should_broadcast_lhs,
                 attrs.should_broadcast_rhs,
                 handle);
}

TaskImplFunction get_element_binary_init_task_impl() {
  return TaskImplFunction{InitOpTaskImplFunction{init_task_impl}};
}

TaskImplFunction get_element_binary_fwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{forward_task_impl}};
}

TaskImplFunction get_element_binary_bwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{backward_task_impl}};
}

}; // namespace FlexFlow
