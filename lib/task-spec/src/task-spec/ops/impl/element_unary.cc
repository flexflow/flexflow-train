#include "task-spec/ops/impl/element_unary.h"
#include "kernels/element_unary_kernels.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "task-spec/profiling.h"
#include "utils/hash-utils.h"

namespace FlexFlow {

using namespace FlexFlow::Kernels::ElementUnary;

static DeviceSpecificPerDeviceOpState
    init_task_impl(TaskArgumentAccessor const &acc) {

  ElementUnaryAttrs attrs = acc.get_op_attrs().require_element_unary();
  DeviceType kernel_device_type = acc.get_kernel_device_type();

  TensorShape input_shape = acc.get_tensor_shape(TensorSlotName::INPUT);
  TensorShape output_shape = acc.get_tensor_shape(TensorSlotName::OUTPUT);

  std::optional<ElementUnaryPerDeviceState> per_device_state =
      init_kernel(kernel_device_type,
                  input_shape,
                  output_shape,
                  attrs);

  return DeviceSpecificPerDeviceOpState{
    acc.make_device_specific(per_device_state),
  };
}

static std::optional<milliseconds_t> forward_task_impl(TaskArgumentAccessor const &acc) {
  auto input = acc.get_tensor<Permissions::RO>(TensorSlotName::INPUT);
  auto output = acc.get_tensor<Permissions::WO>(TensorSlotName::OUTPUT);
  ElementUnaryAttrs attrs = acc.get_op_attrs().require_element_unary();

  device_handle_t handle = acc.get_ff_handle();

  ProfilingSettings profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  ElementUnaryPerDeviceState per_device_state = acc.get_per_device_op_state().require_element_unary().value();

  return profile(forward_kernel,
                 profiling,
                 kernel_device_type,
                 "[ElementUnary] forward_time = {:.2lf}ms\n",
                 per_device_state,
                 attrs,
                 handle,
                 input,
                 output);
}

static std::optional<milliseconds_t>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  auto input = acc.get_tensor<Permissions::RO>(TensorSlotName::INPUT);
  auto input_grad = acc.get_tensor_grad<Permissions::RW>(TensorSlotName::INPUT);
  auto output = acc.get_tensor<Permissions::RO>(TensorSlotName::OUTPUT);
  auto output_grad = acc.get_tensor_grad<Permissions::RO>(TensorSlotName::OUTPUT);

  ElementUnaryAttrs attrs = acc.get_op_attrs().require_element_unary();
  device_handle_t handle = acc.get_ff_handle();

  ProfilingSettings profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  ElementUnaryPerDeviceState per_device_state = acc.get_per_device_op_state().require_element_unary().value();

  return profile(backward_kernel,
                 profiling,
                 kernel_device_type,
                 "[ElementUnary] backward_time = {:.2lf}ms\n",
                 per_device_state,
                 attrs,
                 handle,
                 output,
                 output_grad,
                 input,
                 input_grad);
}

TaskImplFunction get_element_unary_init_task_impl() {
  return TaskImplFunction{InitOpTaskImplFunction{init_task_impl}};
}

TaskImplFunction get_element_unary_fwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{forward_task_impl}};
}

TaskImplFunction get_element_unary_bwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{backward_task_impl}};
}

} // namespace FlexFlow
