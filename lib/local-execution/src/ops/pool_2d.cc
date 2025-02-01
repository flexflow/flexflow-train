#include "pool_2d.h"
#include "kernels/pool_2d_kernels.h"

#include "op-attrs/get_output_shapes.h"
#include "op-attrs/ops/pool_2d.h"
#include "utils/exception.h"
#include "utils/hash-utils.h"

using namespace FlexFlow::Kernels::Pool2D;

namespace FlexFlow {

enum Slots { INPUT, OUTPUT, ATTRS, PROFILING, PER_DEVICE_STATE, HANDLE };

OpTaskInvocation init(Pool2DAttrs const &attrs) {
  OpTaskBinding binding;
  binding.bind(INPUT, input_tensor(0));
  binding.bind(OUTPUT, output_tensor(0));
  binding.bind_arg(ATTRS, attrs);
  binding.bind_arg(HANDLE, ff_handle());

  return {task_id_t::POOL2D_INIT_TASK_ID, binding};
}

static nonnegative_int calculate_padding(nonnegative_int output_size,
                                         nonnegative_int stride,
                                         nonnegative_int kernel_size,
                                         nonnegative_int input_size) {
  int o = output_size.unwrap_nonnegative();
  int s = stride.unwrap_nonnegative();
  int k = kernel_size.unwrap_nonnegative();
  int i = kernel_size.unwrap_nonnegative();

  return nonnegative_int{
      ((o - 1) * s + k - i + 1) / 2,
  };
}

static DeviceSpecificDeviceStates
    init_task_impl(TaskArgumentAccessor const &acc) {
  auto const &attrs = acc.get_argument<Pool2DAttrs>(ATTRS);
  PerDeviceFFHandle handle = acc.get_argument<PerDeviceFFHandle>(HANDLE);

  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

  nonnegative_int input_w = input.shape.at(ff_dim_t{0_n});
  nonnegative_int input_h = input.shape.at(ff_dim_t{1_n});
  nonnegative_int input_c = input.shape.at(ff_dim_t{2_n});
  nonnegative_int input_n = input.shape.at(ff_dim_t{3_n});
  nonnegative_int output_w = output.shape.at(ff_dim_t{0_n});
  nonnegative_int output_h = output.shape.at(ff_dim_t{1_n});
  nonnegative_int output_c = output.shape.at(ff_dim_t{2_n});
  nonnegative_int output_n = output.shape.at(ff_dim_t{3_n});

  Pool2DPerDeviceState per_device_state =
      init_kernel(handle,
                  attrs.activation,
                  input_w.unwrap_nonnegative(),
                  input_h.unwrap_nonnegative(),
                  input_c.unwrap_nonnegative(),
                  input_n.unwrap_nonnegative(),
                  output_w.unwrap_nonnegative(),
                  output_h.unwrap_nonnegative(),
                  output_c.unwrap_nonnegative(),
                  output_n.unwrap_nonnegative(),
                  attrs.padding_h.unwrap_nonnegative(),
                  attrs.padding_w.unwrap_nonnegative(),
                  attrs.kernel_h.unwrap_nonnegative(),
                  attrs.kernel_w.unwrap_nonnegative(),
                  attrs.stride_h.unwrap_nonnegative(),
                  attrs.stride_w.unwrap_nonnegative(),
                  attrs.pool_type);

  return DeviceSpecificDeviceStates{
      DeviceSpecific<Pool2DPerDeviceState>::create(per_device_state)};
}

OpTaskInvocation forward(Pool2DAttrs const &attrs) {
  OpTaskBinding binding;
  binding.bind(INPUT, input_tensor(0));
  binding.bind(OUTPUT, output_tensor(0));

  binding.bind_arg(PROFILING, profiling_settings());
  binding.bind_arg(PER_DEVICE_STATE,
                   per_device_op_state<Pool2DPerDeviceState>());

  return {task_id_t::POOL2D_FWD_TASK_ID, binding};
}

OpTaskInvocation backward(Pool2DAttrs const &attrs) {
  OpTaskBinding b = infer_bwd_binding(forward(attrs).binding);

  return {task_id_t::POOL2D_BWD_TASK_ID, b};
}

static std::optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  Pool2DPerDeviceState state =
      acc.get_argument<Pool2DPerDeviceState>(PER_DEVICE_STATE);

  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

  return profile(forward_kernel,
                 profiling,
                 "[Pool2D] forward_time = {:.2lf}ms\n",
                 state,
                 input.get_float_ptr(),
                 output.get_float_ptr());
}

static std::optional<float>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  Pool2DPerDeviceState state =
      acc.get_argument<Pool2DPerDeviceState>(PER_DEVICE_STATE);

  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto input_grad = acc.get_tensor<Permissions::RW>(INPUT);
  auto output = acc.get_tensor<Permissions::RO>(OUTPUT);
  auto output_grad = acc.get_tensor<Permissions::RO>(OUTPUT);

  return profile(backward_kernel,
                 profiling,
                 "[Pool2D] backward_time = {:.2lf}ms\n",
                 state,
                 input.get_float_ptr(),
                 input_grad.get_float_ptr(),
                 output.get_float_ptr(),
                 output_grad.get_float_ptr());
}

TaskImplFunction get_pool_2d_init_task_impl() {
  return TaskImplFunction{InitTaskImplFunction{init_task_impl}};
}
TaskImplFunction get_pool_2d_fwd_task_impl() {
  return TaskImplFunction{FwdBwdTaskImplFunction{forward_task_impl}};
}
TaskImplFunction get_pool_2d_bwd_task_impl() {
  return TaskImplFunction{FwdBwdTaskImplFunction{backward_task_impl}};
}

OpTaskSignature get_pool_2d_init_signature() {
  OpTaskSignature init(OpTaskType::INIT);

  init.add_input_slot(INPUT);
  init.add_output_slot(OUTPUT);

  init.add_arg_slot<Pool2DAttrs>(ATTRS);
  init.add_unchecked_arg_slot<PerDeviceFFHandle>(HANDLE);

  init.add_return_value<FlexFlow::Pool2DPerDeviceState>();
  return init;
}
OpTaskSignature get_pool_2d_fwd_signature() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT);
  fwd.add_arg_slot<ProfilingSettings>(PROFILING);

  fwd.add_unchecked_arg_slot<Pool2DPerDeviceState>(PER_DEVICE_STATE);
  return fwd;
}
OpTaskSignature get_pool_2d_bwd_signature() {
  OpTaskSignature bwd = infer_bwd_signature(get_pool_2d_fwd_signature());
  return bwd;
}

std::vector<task_id_t> get_task_ids(Pool2DAttrs const &) {
  return {task_id_t::POOL2D_INIT_TASK_ID,
          task_id_t::POOL2D_FWD_TASK_ID,
          task_id_t::POOL2D_BWD_TASK_ID};
}

}; // namespace FlexFlow
