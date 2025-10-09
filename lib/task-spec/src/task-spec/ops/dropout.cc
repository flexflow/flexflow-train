#include "task-spec/ops/dropout.h"
#include "kernels/dropout_kernels.h"
#include "task-spec/op_task_invocation.h"
#include "task-spec/op_task_signature.h"
#include "task-spec/profiling.h"
#include "utils/hash-utils.h"

namespace FlexFlow {

using namespace FlexFlow::Kernels::Dropout;

enum Slots {
  INPUT,
  OUTPUT,
  ATTRS,
  PER_DEVICE_STATE,
  FF_HANDLE,
  PROFILING,
  KERNEL_DEVICE_TYPE
};

OpTaskInvocation init(DropoutAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(ATTRS, attrs);
  binding.bind_arg(FF_HANDLE, ff_handle());
  binding.bind_arg(KERNEL_DEVICE_TYPE, kernel_device_type());

  binding.bind(OUTPUT, output_tensor(0_n));

  return OpTaskInvocation{
      task_id_t::DROPOUT_INIT_TASK_ID,
      binding,
  };
}

OpTaskInvocation forward(DropoutAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind(INPUT, input_tensor(0_n));
  binding.bind(OUTPUT, output_tensor(0_n));

  binding.bind_arg(PROFILING, profiling_settings());
  binding.bind_arg(KERNEL_DEVICE_TYPE, kernel_device_type());
  binding.bind_arg(PER_DEVICE_STATE,
                   per_device_op_state<std::optional<DropoutPerDeviceState>>());

  return OpTaskInvocation{
      task_id_t::DROPOUT_FWD_TASK_ID,
      binding,
  };
}

OpTaskInvocation backward(DropoutAttrs const &attrs) {
  OpTaskBinding b = infer_bwd_binding(forward(attrs).binding);

  return OpTaskInvocation{
      task_id_t::DROPOUT_BWD_TASK_ID,
      b,
  };
}

static DeviceSpecificPerDeviceOpState
    init_task_impl(TaskArgumentAccessor const &acc) {
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);
  Allocator allocator = acc.get_allocator();
  device_handle_t handle = acc.get_argument<device_handle_t>(FF_HANDLE);
  DeviceType kernel_device_type =
      acc.get_argument<DeviceType>(KERNEL_DEVICE_TYPE);
  auto const &attrs = acc.get_argument<DropoutAttrs>(ATTRS);

  std::optional<DropoutPerDeviceState> per_device_state =
      init_kernel(kernel_device_type,
                  handle,
                  attrs.rate,
                  attrs.seed,
                  output.shape,
                  allocator);

  return DeviceSpecificPerDeviceOpState{
    acc.make_device_specific(per_device_state),
  };
}

static std::optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  auto per_device_state =
      acc.get_argument<std::optional<DropoutPerDeviceState>>(PER_DEVICE_STATE);
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  DeviceType kernel_device_type =
      acc.get_argument<DeviceType>(KERNEL_DEVICE_TYPE);
  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

  return profile(forward_kernel,
                 profiling,
                 kernel_device_type,
                 "[Dropout] forward_time = {:.2lf}ms\n",
                 per_device_state,
                 input.get_float_ptr(),
                 output.get_float_ptr());
}

static std::optional<float>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  auto const &attrs = acc.get_argument<DropoutAttrs>(ATTRS);
  auto per_device_state =
      acc.get_argument<DropoutPerDeviceState>(PER_DEVICE_STATE);
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  DeviceType kernel_device_type =
      acc.get_argument<DeviceType>(KERNEL_DEVICE_TYPE);

  auto input_grad = acc.get_tensor_grad<Permissions::RW>(INPUT);
  auto output_grad = acc.get_tensor_grad<Permissions::RO>(OUTPUT);

  return profile(backward_kernel,
                 profiling,
                 kernel_device_type,
                 "[Dropout] backward_time = {:.2lf}ms\n",
                 per_device_state,
                 output_grad.get_float_ptr(),
                 input_grad.get_float_ptr());
}

TaskImplFunction get_dropout_init_task_impl() {
  return TaskImplFunction{InitOpTaskImplFunction{init_task_impl}};
}
TaskImplFunction get_dropout_fwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{forward_task_impl}};
}
TaskImplFunction get_dropout_bwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{backward_task_impl}};
}

OpTaskSignature get_dropout_init_signature() {
  OpTaskSignature init(OpTaskType::INIT);

  init.add_arg_slot<DropoutAttrs>(ATTRS);
  init.add_arg_slot<DeviceType>(KERNEL_DEVICE_TYPE);
  init.add_unchecked_arg_slot<device_handle_t>(FF_HANDLE);
  init.add_output_slot(OUTPUT);

  init.add_return_value<std::optional<DropoutPerDeviceState>>();

  return init;
}

OpTaskSignature get_dropout_fwd_signature() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_unchecked_arg_slot<std::optional<DropoutPerDeviceState>>(
      PER_DEVICE_STATE);
  fwd.add_arg_slot<ProfilingSettings>(PROFILING);
  fwd.add_arg_slot<DeviceType>(KERNEL_DEVICE_TYPE);

  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT);

  return fwd;
}

OpTaskSignature get_dropout_bwd_signature() {
  OpTaskSignature bwd = infer_bwd_signature(get_dropout_fwd_signature());

  return bwd;
}

std::vector<task_id_t> get_task_ids(DropoutAttrs const &) {
  return {task_id_t::DROPOUT_INIT_TASK_ID,
          task_id_t::DROPOUT_FWD_TASK_ID,
          task_id_t::DROPOUT_BWD_TASK_ID};
}

}; // namespace FlexFlow
