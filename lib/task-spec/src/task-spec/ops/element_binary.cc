#include "task-spec/ops/element_binary.h"
#include "kernels/element_binary_kernels.h"
#include "task-spec/device_specific_device_states.h"
#include "task-spec/profiling.h"
#include "task-spec/task_signature_impl.h"
#include "utils/hash-utils.h"

namespace FlexFlow {

using namespace FlexFlow::Kernels::ElementBinary;

enum Slots {
  LHS_INPUT,
  RHS_INPUT,
  OUTPUT,
  PROFILING,
  PER_DEVICE_STATE,
  HANDLE,
  ATTRS,
  KERNEL_DEVICE_TYPE,
};

OpTaskInvocation init(ElementBinaryAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind(LHS_INPUT, input_tensor(0_n));
  binding.bind(RHS_INPUT, input_tensor(1_n));
  binding.bind(OUTPUT, output_tensor(0_n));

  binding.bind_arg(ATTRS, attrs);
  binding.bind_arg(HANDLE, ff_handle());
  binding.bind_arg(KERNEL_DEVICE_TYPE, kernel_device_type());

  return OpTaskInvocation{
      task_id_t::ELEMENTBINARY_INIT_TASK_ID,
      binding,
  };
}

OpTaskInvocation forward(ElementBinaryAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind(LHS_INPUT, input_tensor(0_n));
  binding.bind(RHS_INPUT, input_tensor(1_n));
  binding.bind(OUTPUT, output_tensor(0_n));

  binding.bind_arg(ATTRS, attrs);
  binding.bind_arg(PROFILING, profiling_settings());
  binding.bind_arg(PER_DEVICE_STATE,
                   per_device_op_state<ElementBinaryPerDeviceState>());
  binding.bind_arg(HANDLE, ff_handle());
  binding.bind_arg(KERNEL_DEVICE_TYPE, kernel_device_type());

  return OpTaskInvocation{
      task_id_t::ELEMENTBINARY_FWD_TASK_ID,
      binding,
  };
}

OpTaskInvocation backward(ElementBinaryAttrs const &attrs) {
  OpTaskBinding b = infer_bwd_binding(forward(attrs).binding);

  return OpTaskInvocation{
      task_id_t::ELEMENTBINARY_BWD_TASK_ID,
      b,
  };
}

static std::optional<DeviceSpecificDeviceStates>
    init_task_impl(TaskArgumentAccessor const &acc) {
  auto input_lhs = acc.get_tensor<Permissions::RO>(LHS_INPUT);
  auto input_rhs = acc.get_tensor<Permissions::RO>(RHS_INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

  PerDeviceFFHandle handle = acc.get_argument<PerDeviceFFHandle>(HANDLE);
  DeviceType kernel_device_type =
      acc.get_argument<DeviceType>(KERNEL_DEVICE_TYPE);
  auto const &attrs = acc.get_argument<ElementBinaryAttrs>(ATTRS);

  std::optional<ElementBinaryPerDeviceState> per_device_state =
      init_kernel(kernel_device_type,
                  handle,
                  attrs.type,
                  attrs.should_broadcast_lhs,
                  attrs.should_broadcast_rhs,
                  input_lhs.shape,
                  input_rhs.shape,
                  output.shape);
  return make_device_specific_state(per_device_state);
}

static std::optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  DeviceType kernel_device_type =
      acc.get_argument<DeviceType>(KERNEL_DEVICE_TYPE);
  auto per_device_state =
      acc.get_argument<ElementBinaryPerDeviceState>(PER_DEVICE_STATE);
  auto const &attrs = acc.get_argument<ElementBinaryAttrs>(ATTRS);

  auto input_lhs = acc.get_tensor<Permissions::RO>(LHS_INPUT);
  auto input_rhs = acc.get_tensor<Permissions::RO>(RHS_INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);
  PerDeviceFFHandle handle = acc.get_argument<PerDeviceFFHandle>(HANDLE);

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

static std::optional<float>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  auto per_device_state =
      acc.get_argument<ElementBinaryPerDeviceState>(PER_DEVICE_STATE);
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  DeviceType kernel_device_type =
      acc.get_argument<DeviceType>(KERNEL_DEVICE_TYPE);
  auto const &attrs = acc.get_argument<ElementBinaryAttrs>(ATTRS);
  PerDeviceFFHandle handle = acc.get_argument<PerDeviceFFHandle>(HANDLE);

  auto input_lhs = acc.get_tensor<Permissions::RO>(LHS_INPUT);
  auto input_rhs = acc.get_tensor<Permissions::RO>(RHS_INPUT);

  auto output_grad = acc.get_tensor_grad<Permissions::RO>(OUTPUT);
  auto input_lhs_grad = acc.get_tensor_grad<Permissions::RW>(LHS_INPUT);
  auto input_rhs_grad = acc.get_tensor_grad<Permissions::RW>(RHS_INPUT);

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

OpTaskSignature get_element_binary_init_signature() {
  OpTaskSignature init(OpTaskType::INIT);

  init.add_input_slot(LHS_INPUT);
  init.add_input_slot(RHS_INPUT);
  init.add_output_slot(OUTPUT);

  init.add_arg_slot<BatchMatmulAttrs>(ATTRS);
  init.add_arg_slot<DeviceType>(KERNEL_DEVICE_TYPE);
  init.add_unchecked_arg_slot<PerDeviceFFHandle>(HANDLE);

  init.add_return_value<ElementBinaryPerDeviceState>();

  return init;
}

OpTaskSignature get_element_binary_fwd_signature() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_arg_slot<ProfilingSettings>(PROFILING);
  fwd.add_unchecked_arg_slot<ElementBinaryPerDeviceState>(PER_DEVICE_STATE);
  fwd.add_arg_slot<ElementBinaryAttrs>(ATTRS);
  fwd.add_arg_slot<DeviceType>(KERNEL_DEVICE_TYPE);
  fwd.add_unchecked_arg_slot<PerDeviceFFHandle>(HANDLE);

  fwd.add_input_slot(LHS_INPUT);
  fwd.add_input_slot(RHS_INPUT);
  fwd.add_output_slot(OUTPUT);

  return fwd;
}

OpTaskSignature get_element_binary_bwd_signature() {
  OpTaskSignature bwd = infer_bwd_signature(get_element_binary_fwd_signature());

  return bwd;
}

std::vector<task_id_t> get_task_ids(ElementBinaryAttrs const &) {
  return {task_id_t::ELEMENTBINARY_INIT_TASK_ID,
          task_id_t::ELEMENTBINARY_FWD_TASK_ID,
          task_id_t::ELEMENTBINARY_BWD_TASK_ID};
}

}; // namespace FlexFlow
