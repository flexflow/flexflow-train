#include "task-spec/ops/linear.h"
#include "kernels/format_accessor_contents.h"
#include "kernels/linear_kernels.h"
#include "op-attrs/ff_dim_t.h"
#include "task-spec/profiling.h"
#include "task-spec/task_argument_accessor.h"
#include "utils/exception.h"
#include "utils/hash-utils.h"

namespace FlexFlow {

enum slots {
  INPUT,
  OUTPUT,
  WEIGHT,
  BIAS,
  ATTRS,
  PROFILING,
  HANDLE,
  PER_DEVICE_STATE,
  KERNEL_DEVICE_TYPE,
};

OpTaskInvocation init(LinearAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(HANDLE, ff_handle());
  binding.bind_arg(KERNEL_DEVICE_TYPE, kernel_device_type());
  binding.bind_arg(ATTRS, attrs);

  binding.bind(INPUT, input_tensor(0_n));
  binding.bind(WEIGHT, weight_tensor(0_n));
  binding.bind(OUTPUT, output_tensor(0_n));

  return OpTaskInvocation{
      task_id_t::LINEAR_INIT_TASK_ID,
      binding,
  };
}

OpTaskInvocation forward(LinearAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind(INPUT, input_tensor(0_n));
  binding.bind(WEIGHT, weight_tensor(0_n));
  binding.bind(OUTPUT, output_tensor(0_n));
  if (attrs.use_bias) {
    binding.bind(BIAS, weight_tensor(1_n));
  }

  binding.bind_arg(PROFILING, profiling_settings());
  binding.bind_arg(KERNEL_DEVICE_TYPE, kernel_device_type());
  binding.bind_arg(PER_DEVICE_STATE,
                   per_device_op_state<std::optional<LinearPerDeviceState>>());
  binding.bind_arg(ATTRS, attrs);

  return OpTaskInvocation{
      task_id_t::LINEAR_FWD_TASK_ID,
      binding,
  };
}

OpTaskInvocation backward(LinearAttrs const &attrs) {
  OpTaskBinding b = infer_bwd_binding(forward(attrs).binding);

  return OpTaskInvocation{
      task_id_t::LINEAR_BWD_TASK_ID,
      b,
  };
}

static DeviceSpecificPerDeviceOpState
    init_task_impl(TaskArgumentAccessor const &acc) {
  auto const &attrs = acc.get_argument<LinearAttrs>(ATTRS);
  device_handle_t handle = acc.get_argument<device_handle_t>(HANDLE);
  DeviceType kernel_device_type =
      acc.get_argument<DeviceType>(KERNEL_DEVICE_TYPE);

  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto weight = acc.get_tensor<Permissions::RO>(WEIGHT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);
  positive_int out_dim = dim_at_idx(output.shape.dims, ff_dim_t{0_n});
  positive_int batch_size = dim_at_idx(output.shape.dims, ff_dim_t{1_n});

  std::optional<LinearPerDeviceState> per_device_state =
      linear_init_kernel(kernel_device_type,
                         handle,
                         attrs.activation,
                         attrs.regularizer,
                         attrs.use_bias,
                         input.shape.data_type,
                         weight.shape.data_type,
                         output.shape.data_type,
                         batch_size.int_from_positive_int(),
                         attrs.out_channels.int_from_positive_int());

  return DeviceSpecificPerDeviceOpState{
    acc.make_device_specific(per_device_state),
  };
}

static std::optional<milliseconds_t> forward_task_impl(TaskArgumentAccessor const &acc) {
  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto weight = acc.get_tensor<Permissions::RO>(WEIGHT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

  auto per_device_state =
      acc.get_argument<std::optional<LinearPerDeviceState>>(PER_DEVICE_STATE);
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  DeviceType kernel_device_type =
      acc.get_argument<DeviceType>(KERNEL_DEVICE_TYPE);
  auto attrs = acc.get_argument<LinearAttrs>(ATTRS);

  std::optional<GenericTensorAccessorR> bias = std::nullopt;
  if (attrs.use_bias) {
    bias = acc.get_tensor<Permissions::RO>(BIAS);
  }

  auto result = profile(linear_forward_kernel,
                        profiling,
                        kernel_device_type,
                        "[Linear] forward_time = {:.2lf}ms\n",
                        per_device_state,
                        attrs,
                        input,
                        output,
                        weight,
                        bias);

  return result;
}

static std::optional<milliseconds_t>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto weight = acc.get_tensor<Permissions::RO>(WEIGHT);
  auto output = acc.get_tensor<Permissions::RO>(OUTPUT);

  auto input_grad = acc.get_tensor_grad<Permissions::RW>(INPUT);
  auto weight_grad = acc.get_tensor_grad<Permissions::RW>(WEIGHT);
  auto output_grad = acc.get_tensor_grad<Permissions::RW>(OUTPUT);

  auto per_device_state =
      acc.get_argument<std::optional<LinearPerDeviceState>>(PER_DEVICE_STATE);
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  DeviceType kernel_device_type =
      acc.get_argument<DeviceType>(KERNEL_DEVICE_TYPE);
  auto attrs = acc.get_argument<LinearAttrs>(ATTRS);

  std::optional<GenericTensorAccessorW> bias_grad = std::nullopt;
  if (attrs.use_bias) {
    bias_grad = acc.get_tensor<Permissions::RW>(BIAS);
  }

  auto result = profile(linear_backward_kernel,
                        profiling,
                        kernel_device_type,
                        "[Linear] backward_time = {:.2lf}ms\n",
                        per_device_state,
                        attrs,
                        output,
                        output_grad,
                        input,
                        input_grad,
                        weight,
                        weight_grad,
                        bias_grad);

  return result;
}

TaskImplFunction get_linear_init_task_impl() {
  return TaskImplFunction{InitOpTaskImplFunction{init_task_impl}};
}

TaskImplFunction get_linear_fwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{forward_task_impl}};
}

TaskImplFunction get_linear_bwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{backward_task_impl}};
}

OpTaskSignature get_linear_init_signature() {
  OpTaskSignature init(OpTaskType::INIT);

  init.add_input_slot(INPUT);
  init.add_weight_slot(WEIGHT);
  init.add_output_slot(OUTPUT);

  init.add_arg_slot<LinearAttrs>(ATTRS);
  init.add_arg_slot<DeviceType>(KERNEL_DEVICE_TYPE);
  init.add_unchecked_arg_slot<PerDeviceFFHandle>(HANDLE);

  init.add_return_value<LinearPerDeviceState>();
  return init;
}

OpTaskSignature get_linear_fwd_signature() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_input_slot(INPUT);
  fwd.add_weight_slot(WEIGHT);
  fwd.add_optional_weight_slot(BIAS);
  fwd.add_output_slot(OUTPUT);

  fwd.add_arg_slot<ProfilingSettings>(PROFILING);
  fwd.add_arg_slot<DeviceType>(KERNEL_DEVICE_TYPE);
  fwd.add_arg_slot<LinearAttrs>(ATTRS);
  fwd.add_unchecked_arg_slot<LinearPerDeviceState>(PER_DEVICE_STATE);
  return fwd;
}

OpTaskSignature get_linear_bwd_signature() {
  OpTaskSignature bwd = infer_bwd_signature(get_linear_fwd_signature());
  return bwd;
}

std::unordered_set<task_id_t> get_task_ids(LinearAttrs const &) {
  return {task_id_t::LINEAR_INIT_TASK_ID,
          task_id_t::LINEAR_FWD_TASK_ID,
          task_id_t::LINEAR_BWD_TASK_ID};
}

}; // namespace FlexFlow
