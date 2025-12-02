/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "task-spec/ops/impl/batch_norm.h"
#include "kernels/batch_norm_kernels.h"
#include "task-spec/profiling.h"

namespace FlexFlow {

using namespace FlexFlow::Kernels::BatchNorm;

enum Slots {
  INPUT,
  SCALE,
  BIAS,
  OUTPUT,
  ATTRS,
  PROFILING,
  PER_DEVICE_STATE,
  RELU,
  HANDLE,
  KERNEL_DEVICE_TYPE,
};

OpTaskInvocation init(BatchNormAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind(INPUT, input_tensor(0_n));
  binding.bind(BIAS, weight_tensor(1_n));
  binding.bind(OUTPUT, output_tensor(0_n));

  binding.bind_arg(ATTRS, attrs);
  binding.bind_arg(PROFILING, profiling_settings());
  binding.bind_arg(HANDLE, ff_handle());
  binding.bind_arg(KERNEL_DEVICE_TYPE, kernel_device_type());

  return OpTaskInvocation{
      op_task_id_t::INIT,
      binding,
  };
}

OpTaskInvocation forward(BatchNormAttrs const &attrs) {
  OpTaskBinding binding;
  binding.bind_arg(PROFILING, profiling_settings());
  binding.bind_arg(KERNEL_DEVICE_TYPE, kernel_device_type());
  binding.bind_arg(
      PER_DEVICE_STATE,
      per_device_op_state<std::optional<BatchNormPerDeviceState>>());

  binding.bind(INPUT, input_tensor(0_n));
  binding.bind(SCALE, weight_tensor(0_n));
  binding.bind(BIAS, weight_tensor(1_n));
  binding.bind(OUTPUT, output_tensor(0_n));

  return OpTaskInvocation{
      op_task_id_t::FWD,
      binding,
  };
}

OpTaskInvocation backward(BatchNormAttrs const &attrs) {
  OpTaskBinding binding = infer_bwd_binding(forward(attrs).binding);

  return OpTaskInvocation{
      op_task_id_t::BWD,
      binding,
  };
}

static DeviceSpecificPerDeviceOpState
    init_task_impl(TaskArgumentAccessor const &acc) {
  Allocator allocator = acc.get_allocator();
  device_handle_t handle = acc.get_argument<device_handle_t>(HANDLE);
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  DeviceType kernel_device_type =
      acc.get_argument<DeviceType>(KERNEL_DEVICE_TYPE);

  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);
  auto const &attrs = acc.get_argument<BatchNormAttrs>(ATTRS);

  positive_int output_w = dim_at_idx(output.shape.dims, legion_dim_t{0_n});
  positive_int output_h = dim_at_idx(output.shape.dims, legion_dim_t{1_n});
  positive_int output_c = dim_at_idx(output.shape.dims, legion_dim_t{2_n});
  positive_int output_n = dim_at_idx(output.shape.dims, legion_dim_t{3_n});

  float *runningMean;

  std::optional<BatchNormPerDeviceState> per_device_state = init_kernel(
      /*device_type=*/kernel_device_type,
      /*handle=*/handle,
      /*allocator=*/allocator,
      /*runningMean=*/runningMean,
      /*output_n=*/output_n.int_from_positive_int(),
      /*output_c=*/output_c.int_from_positive_int(),
      /*output_h=*/output_h.int_from_positive_int(),
      /*output_w=*/output_w.int_from_positive_int(),
      /*relu=*/attrs.relu);

  return DeviceSpecificPerDeviceOpState{
    acc.make_device_specific(per_device_state),
  };
}

static std::optional<milliseconds_t> forward_task_impl(TaskArgumentAccessor const &acc) {
  auto per_device_state =
      acc.get_argument<BatchNormPerDeviceState>(PER_DEVICE_STATE);
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  DeviceType kernel_device_type =
      acc.get_argument<DeviceType>(KERNEL_DEVICE_TYPE);

  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);
  auto scale = acc.get_tensor<Permissions::RO>(SCALE);
  auto bias = acc.get_tensor<Permissions::RO>(SCALE);

  return profile(forward_kernel,
                 profiling,
                 kernel_device_type,
                 "[BatchNorm] forward_time = {:.2lf}ms\n",
                 per_device_state,
                 input.get_float_ptr(),
                 output.get_float_ptr(),
                 scale.get_float_ptr(),
                 bias.get_float_ptr());
}

static std::optional<milliseconds_t>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  auto per_device_state =
      acc.get_argument<BatchNormPerDeviceState>(PER_DEVICE_STATE);
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  DeviceType kernel_device_type =
      acc.get_argument<DeviceType>(KERNEL_DEVICE_TYPE);

  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto input_grad = acc.get_tensor_grad<Permissions::RW>(INPUT);
  auto output = acc.get_tensor<Permissions::RO>(OUTPUT);
  auto output_grad = acc.get_tensor_grad<Permissions::RW>(OUTPUT);
  auto scale = acc.get_tensor<Permissions::RO>(SCALE);
  auto scale_grad = acc.get_tensor_grad<Permissions::RW>(SCALE);
  auto bias_grad = acc.get_tensor_grad<Permissions::RW>(BIAS);

  return profile(backward_kernel,
                 profiling,
                 kernel_device_type,
                 "[BatchNorm] backward_time = {:.2lf}ms\n",
                 per_device_state,
                 output.get_float_ptr(),
                 output_grad.get_float_ptr(),
                 input.get_float_ptr(),
                 input_grad.get_float_ptr(),
                 scale.get_float_ptr(),
                 scale_grad.get_float_ptr(),
                 bias_grad.get_float_ptr(),
                 get_num_elements(output.shape.dims).int_from_positive_int());
}

TaskImplFunction get_batch_norm_init_task_impl() {
  return TaskImplFunction{InitOpTaskImplFunction{init_task_impl}};
}

TaskImplFunction get_batch_norm_fwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{forward_task_impl}};
}

TaskImplFunction get_batch_norm_bwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{backward_task_impl}};
}

}; // namespace FlexFlow
