#include "task-spec/ops/conv_2d.h"
#include "kernels/conv_2d_kernels.h"
#include "task-spec/device_specific_device_states.h"
#include "task-spec/profiling.h"

namespace FlexFlow {

using namespace FlexFlow::Kernels::Conv2D;

enum Slots {
  INPUT,
  OUTPUT,
  FILTER,
  BIAS,
  ATTRS,
  PROFILING,
  PER_DEVICE_STATE,
  HANDLE,
  KERNEL_DEVICE_TYPE,
};

OpTaskInvocation init(Conv2DAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind(INPUT, input_tensor(0_n));
  binding.bind(OUTPUT, output_tensor(0_n));
  binding.bind(FILTER, weight_tensor(0_n));
  binding.bind_arg(ATTRS, attrs);
  binding.bind_arg(HANDLE, ff_handle());
  binding.bind_arg(KERNEL_DEVICE_TYPE, kernel_device_type());

  return OpTaskInvocation{
      task_id_t::CONV2D_INIT_TASK_ID,
      binding,
  };
}

OpTaskInvocation forward(Conv2DAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(ATTRS, attrs);
  binding.bind_arg(PROFILING, profiling_settings());
  binding.bind_arg(KERNEL_DEVICE_TYPE, kernel_device_type());
  binding.bind_arg(PER_DEVICE_STATE,
                   per_device_op_state<Conv2DPerDeviceState>());

  binding.bind(INPUT, input_tensor(0_n));
  binding.bind(OUTPUT, output_tensor(0_n));
  binding.bind(FILTER, weight_tensor(0_n));
  binding.bind(BIAS, weight_tensor(1_n));

  return OpTaskInvocation{
      task_id_t::CONV2D_FWD_TASK_ID,
      binding,
  };
}

OpTaskInvocation backward(Conv2DAttrs const &attrs) {
  OpTaskBinding binding = infer_bwd_binding(forward(attrs).binding);

  return OpTaskInvocation{
      task_id_t::CONV2D_BWD_TASK_ID,
      binding,
  };
}

static std::optional<DeviceSpecificDeviceStates>
    init_task_impl(TaskArgumentAccessor const &acc) {

  PerDeviceFFHandle handle = acc.get_argument<PerDeviceFFHandle>(HANDLE);
  DeviceType kernel_device_type =
      acc.get_argument<DeviceType>(KERNEL_DEVICE_TYPE);
  auto attrs = acc.get_argument<Conv2DAttrs>(ATTRS);
  auto input = acc.get_tensor<Permissions::WO>(INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);
  auto filter = acc.get_tensor<Permissions::RO>(FILTER);
  auto filter_grad = acc.get_tensor_grad<Permissions::RW>(FILTER);

  std::optional<Conv2DPerDeviceState> per_device_state = init_kernel(
      /*device_type=*/kernel_device_type,
      /*handle=*/handle,
      /*activation=*/attrs.activation,
      /*kernel_h=*/attrs.kernel_h.int_from_positive_int(),
      /*kernel_w=*/attrs.kernel_w.int_from_positive_int(),
      /*groups=*/attrs.groups.int_from_positive_int(),
      /*padding_h=*/attrs.padding_h.unwrap_nonnegative(),
      /*padding_w=*/attrs.padding_w.unwrap_nonnegative(),
      /*stride_h=*/attrs.stride_h.int_from_positive_int(),
      /*stride_w=*/attrs.stride_w.int_from_positive_int(),
      /*input=*/input,
      /*output=*/output,
      /*filter_ptr=*/filter.get_float_ptr(),
      /*filter_grad_ptr=*/filter_grad.get_float_ptr());
  return make_device_specific_state(per_device_state);
}

static std::optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  DeviceType kernel_device_type =
      acc.get_argument<DeviceType>(KERNEL_DEVICE_TYPE);
  auto per_device_state =
      acc.get_argument<Conv2DPerDeviceState>(PER_DEVICE_STATE);
  auto attrs = acc.get_argument<Conv2DAttrs>(ATTRS);

  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto filter = acc.get_tensor<Permissions::RO>(FILTER);
  auto bias = acc.get_tensor<Permissions::RO>(BIAS);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

  return profile(forward_kernel,
                 profiling,
                 kernel_device_type,
                 "[Conv2d] forward_time = {:.2lf}ms\n",
                 per_device_state,
                 input.get_float_ptr(),
                 output.get_float_ptr(),
                 filter.get_float_ptr(),
                 bias.get_float_ptr(),
                 attrs.activation);
}

static std::optional<float>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  DeviceType kernel_device_type =
      acc.get_argument<DeviceType>(KERNEL_DEVICE_TYPE);
  auto per_device_state =
      acc.get_argument<Conv2DPerDeviceState>(PER_DEVICE_STATE);
  auto attrs = acc.get_argument<Conv2DAttrs>(ATTRS);

  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);
  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto filter = acc.get_tensor<Permissions::RO>(FILTER);

  auto input_grad = acc.get_tensor_grad<Permissions::RW>(INPUT);
  auto output_grad = acc.get_tensor_grad<Permissions::RW>(OUTPUT);
  auto filter_grad = acc.get_tensor_grad<Permissions::RW>(FILTER);
  auto bias_grad = acc.get_tensor_grad<Permissions::RW>(BIAS);

  return profile(backward_kernel,
                 profiling,
                 kernel_device_type,
                 "[Conv2d] backward_time = {:.2lf}ms\n",
                 per_device_state,
                 output.get_float_ptr(),
                 output_grad.get_float_ptr(),
                 input.get_float_ptr(),
                 input_grad.get_float_ptr(),
                 filter.get_float_ptr(),
                 filter_grad.get_float_ptr(),
                 bias_grad.get_float_ptr(),
                 attrs.activation);
}

TaskImplFunction get_conv_2d_init_task_impl() {
  return TaskImplFunction{InitOpTaskImplFunction{init_task_impl}};
}
TaskImplFunction get_conv_2d_fwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{forward_task_impl}};
}
TaskImplFunction get_conv_2d_bwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{backward_task_impl}};
}

OpTaskSignature get_conv_2d_init_signature() {
  OpTaskSignature init(OpTaskType::INIT);

  init.add_input_slot(INPUT);
  init.add_output_slot(OUTPUT);
  init.add_weight_slot(FILTER);
  init.add_arg_slot<Conv2DAttrs>(ATTRS);
  init.add_arg_slot<DeviceType>(KERNEL_DEVICE_TYPE);
  init.add_unchecked_arg_slot<PerDeviceFFHandle>(HANDLE);

  init.add_return_value<Conv2DPerDeviceState>();

  return init;
}

OpTaskSignature get_conv_2d_fwd_signature() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_arg_slot<bool>(PROFILING);
  fwd.add_unchecked_arg_slot<Conv2DPerDeviceState>(PER_DEVICE_STATE);
  fwd.add_arg_slot<DeviceType>(KERNEL_DEVICE_TYPE);
  fwd.add_arg_slot<Conv2DAttrs>(ATTRS);

  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT);
  fwd.add_weight_slot(FILTER);
  fwd.add_weight_slot(BIAS);

  return fwd;
}

OpTaskSignature get_conv_2d_bwd_signature() {
  OpTaskSignature bwd = infer_bwd_signature(get_conv_2d_fwd_signature());

  return bwd;
}

std::vector<task_id_t> get_task_ids(Conv2DAttrs const &) {
  return {task_id_t::CONV2D_INIT_TASK_ID,
          task_id_t::CONV2D_FWD_TASK_ID,
          task_id_t::CONV2D_BWD_TASK_ID};
}

} // namespace FlexFlow
