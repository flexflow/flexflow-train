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

#include "task-spec/ops/impl/softmax.h"
#include "kernels/softmax_kernels.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "task-spec/profiling.h"
#include "utils/exception.h"
#include "utils/hash-utils.h"

namespace FlexFlow {
using namespace FlexFlow::Kernels::Softmax;

enum Slots {
  INPUT,
  OUTPUT,
  ATTRS,
  PROFILING,
  PER_DEVICE_STATE,
  HANDLE,
  KERNEL_DEVICE_TYPE
};

OpTaskInvocation init(SoftmaxAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(HANDLE, ff_handle());
  binding.bind_arg(ATTRS, attrs);
  binding.bind_arg(KERNEL_DEVICE_TYPE, kernel_device_type());

  return OpTaskInvocation{
      op_task_id_t::INIT,
      binding,
  };
}

OpTaskInvocation forward(SoftmaxAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(PER_DEVICE_STATE,
                   per_device_op_state<std::optional<SoftmaxPerDeviceState>>());
  binding.bind_arg(PROFILING, profiling_settings());
  binding.bind_arg(KERNEL_DEVICE_TYPE, kernel_device_type());

  binding.bind(INPUT, input_tensor(0_n));
  binding.bind(OUTPUT, output_tensor(0_n));

  return OpTaskInvocation{
      op_task_id_t::FWD,
      binding,
  };
}

OpTaskInvocation backward(SoftmaxAttrs const &attrs) {
  OpTaskBinding binding = infer_bwd_binding(forward(attrs).binding);

  return OpTaskInvocation{
      op_task_id_t::BWD,
      binding,
  };
}

static DeviceSpecificPerDeviceOpState
    init_task_impl(TaskArgumentAccessor const &acc) {
  device_handle_t handle = acc.get_argument<device_handle_t>(HANDLE);
  DeviceType kernel_device_type =
      acc.get_argument<DeviceType>(KERNEL_DEVICE_TYPE);

  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);
  auto const &attrs = acc.get_argument<SoftmaxAttrs>(ATTRS);

  positive_int output_w = dim_at_idx(output.shape.dims, legion_dim_t{0_n});
  positive_int output_h = dim_at_idx(output.shape.dims, legion_dim_t{1_n});
  positive_int output_c = dim_at_idx(output.shape.dims, legion_dim_t{2_n});
  positive_int output_n = dim_at_idx(output.shape.dims, legion_dim_t{3_n});

  std::optional<SoftmaxPerDeviceState> per_device_state =
      init_kernel(kernel_device_type,
                  handle,
                  attrs.dim,
                  output_n.int_from_positive_int(),
                  output_c.int_from_positive_int(),
                  output_h.int_from_positive_int(),
                  output_w.int_from_positive_int());

  return DeviceSpecificPerDeviceOpState{
    acc.make_device_specific(per_device_state),
  };
}

static std::optional<milliseconds_t> forward_task_impl(TaskArgumentAccessor const &acc) {
  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  DeviceType kernel_device_type =
      acc.get_argument<DeviceType>(KERNEL_DEVICE_TYPE);
  auto per_device_state =
      acc.get_argument<SoftmaxPerDeviceState>(PER_DEVICE_STATE);

  return profile(forward_kernel,
                 profiling,
                 kernel_device_type,
                 "[Softmax] forward_time = {:.2lf}ms\n",
                 per_device_state,
                 input.get_float_ptr(),
                 output.get_float_ptr());
}

static std::optional<milliseconds_t>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  DeviceType kernel_device_type =
      acc.get_argument<DeviceType>(KERNEL_DEVICE_TYPE);

  auto input_grad = acc.get_tensor_grad<Permissions::RW>(INPUT);
  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  assert(input_grad.shape == input.shape);

  auto output_grad = acc.get_tensor_grad<Permissions::RO>(OUTPUT);
  auto output = acc.get_tensor<Permissions::RO>(OUTPUT);

  assert(output_grad.shape == output.shape);

  return profile(
      backward_kernel,
      profiling,
      kernel_device_type,
      "[Softmax] backward_time = {:.2lf}ms\n",
      output_grad.get_float_ptr(),
      input_grad.get_float_ptr(),
      get_num_elements(output_grad.shape.dims).int_from_positive_int());
}

TaskImplFunction get_softmax_init_task_impl() {
  return TaskImplFunction{InitOpTaskImplFunction{init_task_impl}};
}

TaskImplFunction get_softmax_fwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{forward_task_impl}};
}

TaskImplFunction get_softmax_bwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{backward_task_impl}};
}

OpTaskSignature get_softmax_init_signature() {
  OpTaskSignature init(OpTaskType::INIT);

  init.add_unchecked_arg_slot<device_handle_t>(HANDLE);
  init.add_arg_slot<SoftmaxAttrs>(ATTRS);
  init.add_arg_slot<DeviceType>(KERNEL_DEVICE_TYPE);
  init.add_return_value<SoftmaxPerDeviceState>();
  return init;
}

OpTaskSignature get_softmax_fwd_signature() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_arg_slot<ProfilingSettings>(PROFILING);
  fwd.add_arg_slot<DeviceType>(KERNEL_DEVICE_TYPE);
  fwd.add_unchecked_arg_slot<SoftmaxPerDeviceState>(PER_DEVICE_STATE);

  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT);
  return fwd;
}

OpTaskSignature get_softmax_bwd_signature() {
  OpTaskSignature bwd = infer_bwd_signature(get_softmax_fwd_signature());
  return bwd;
}

}; // namespace FlexFlow
