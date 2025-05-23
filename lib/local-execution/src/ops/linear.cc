#include "linear.h"
#include "kernels/linear_kernels.h"
#include "local-execution/task_argument_accessor.h"
#include "op-attrs/ff_dim_t.h"
#include "utils/exception.h"
#include "utils/hash-utils.h"

namespace FlexFlow {

using namespace FlexFlow::Kernels::Linear;

enum slots {
  INPUT,
  OUTPUT,
  WEIGHT,
  BIAS,
  ATTRS,
  PROFILING,
  HANDLE,
  PER_DEVICE_STATE
};

OpTaskInvocation init(LinearAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(HANDLE, ff_handle());
  binding.bind_arg(ATTRS, attrs);

  binding.bind(INPUT, input_tensor(0));
  binding.bind(WEIGHT, weight_tensor(0));
  binding.bind(OUTPUT, output_tensor(0));

  return {task_id_t::LINEAR_INIT_TASK_ID, binding};
}

OpTaskInvocation forward(LinearAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind(INPUT, input_tensor(0));
  binding.bind(WEIGHT, weight_tensor(0));
  binding.bind(OUTPUT, output_tensor(0));
  if (attrs.use_bias) {
    binding.bind(BIAS, weight_tensor(1));
  }

  binding.bind_arg(PROFILING, profiling_settings());
  binding.bind_arg(PER_DEVICE_STATE,
                   per_device_op_state<LinearPerDeviceState>());
  binding.bind_arg(ATTRS, attrs);

  return {task_id_t::LINEAR_FWD_TASK_ID, binding};
}

OpTaskInvocation backward(LinearAttrs const &attrs) {
  OpTaskBinding b = infer_bwd_binding(forward(attrs).binding);

  return {task_id_t::LINEAR_BWD_TASK_ID, b};
}

static DeviceSpecificDeviceStates
    init_task_impl(TaskArgumentAccessor const &acc) {
  auto const &attrs = acc.get_argument<LinearAttrs>(ATTRS);
  PerDeviceFFHandle handle = acc.get_argument<PerDeviceFFHandle>(HANDLE);

  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto weight = acc.get_tensor<Permissions::RO>(WEIGHT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);
  nonnegative_int out_dim = output.shape.at(ff_dim_t{0_n});
  nonnegative_int batch_size = output.shape.at(ff_dim_t{1_n});

  float *one_ptr;

  LinearPerDeviceState per_device_state =
      init_kernel(handle,
                  one_ptr,
                  attrs.activation,
                  attrs.regularizer,
                  attrs.use_bias,
                  input.data_type,
                  weight.data_type,
                  output.data_type,
                  batch_size.unwrap_nonnegative(),
                  attrs.out_channels.unwrap_nonnegative());
  return DeviceSpecificDeviceStates{
      DeviceSpecific<LinearPerDeviceState>::create(per_device_state)};
}

static std::optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto weight = acc.get_tensor<Permissions::RO>(WEIGHT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);
  auto bias = acc.get_tensor<Permissions::RO>(BIAS);

  auto per_device_state =
      acc.get_argument<LinearPerDeviceState>(PER_DEVICE_STATE);
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  auto attrs = acc.get_argument<LinearAttrs>(ATTRS);

  nonnegative_int in_dim = input.shape.at(ff_dim_t{0_n});
  nonnegative_int out_dim = output.shape.at(ff_dim_t{0_n});
  nonnegative_int batch_size = output.shape.get_volume() / out_dim;

  float const *bias_ptr = NULL;
  if (attrs.use_bias) {
    bias_ptr = bias.get_float_ptr();
  }

  return profile(forward_kernel,
                 profiling,
                 "[Linear] forward_time = {:.2lf}ms\n",
                 per_device_state,
                 input.get_float_ptr(),
                 output.get_float_ptr(),
                 weight.get_float_ptr(),
                 bias_ptr,
                 in_dim.unwrap_nonnegative(),
                 out_dim.unwrap_nonnegative(),
                 batch_size.unwrap_nonnegative());
}

;

static std::optional<float>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto weight = acc.get_tensor<Permissions::RO>(WEIGHT);
  auto output = acc.get_tensor<Permissions::RO>(OUTPUT);

  auto input_grad = acc.get_tensor_grad<Permissions::RW>(INPUT);
  auto weight_grad = acc.get_tensor_grad<Permissions::RW>(WEIGHT);
  auto output_grad = acc.get_tensor_grad<Permissions::RW>(OUTPUT);

  auto per_device_state =
      acc.get_argument<LinearPerDeviceState>(PER_DEVICE_STATE);
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  auto attrs = acc.get_argument<LinearAttrs>(ATTRS);

  float *bias_grad_ptr = NULL;
  if (attrs.use_bias) {
    auto bias_grad = acc.get_tensor_grad<Permissions::RW>(BIAS);
    bias_grad_ptr = bias_grad.get_float_ptr();
  }

  nonnegative_int in_dim = input.shape.at(ff_dim_t{0_n});
  nonnegative_int out_dim = output.shape.at(ff_dim_t{0_n});
  nonnegative_int batch_size = output.shape.get_volume() / out_dim;

  return profile(backward_kernel,
                 profiling,
                 "[Linear] backward_time = {:.2lf}ms\n",
                 per_device_state,
                 output.get_float_ptr(),
                 output_grad.get_float_ptr(),
                 input.get_float_ptr(),
                 input_grad.get_float_ptr(),
                 weight.get_float_ptr(),
                 weight_grad.get_float_ptr(),
                 bias_grad_ptr,
                 in_dim.unwrap_nonnegative(),
                 out_dim.unwrap_nonnegative(),
                 batch_size.unwrap_nonnegative());
}

TaskImplFunction get_linear_init_task_impl() {
  return TaskImplFunction{InitTaskImplFunction{init_task_impl}};
}
TaskImplFunction get_linear_fwd_task_impl() {
  return TaskImplFunction{FwdBwdTaskImplFunction{forward_task_impl}};
}
TaskImplFunction get_linear_bwd_task_impl() {
  return TaskImplFunction{FwdBwdTaskImplFunction{backward_task_impl}};
}

OpTaskSignature get_linear_init_signature() {
  OpTaskSignature init(OpTaskType::INIT);

  init.add_input_slot(INPUT);
  init.add_weight_slot(WEIGHT);
  init.add_output_slot(OUTPUT);

  init.add_arg_slot<LinearAttrs>(ATTRS);
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
  fwd.add_arg_slot<LinearAttrs>(ATTRS);
  fwd.add_unchecked_arg_slot<LinearPerDeviceState>(PER_DEVICE_STATE);
  return fwd;
}

OpTaskSignature get_linear_bwd_signature() {
  OpTaskSignature bwd = infer_bwd_signature(get_linear_fwd_signature());
  return bwd;
}

std::vector<task_id_t> get_task_ids(LinearAttrs const &) {
  return {task_id_t::LINEAR_INIT_TASK_ID,
          task_id_t::LINEAR_FWD_TASK_ID,
          task_id_t::LINEAR_BWD_TASK_ID};
}

}; // namespace FlexFlow
