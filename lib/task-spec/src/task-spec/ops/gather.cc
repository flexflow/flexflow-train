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

#include "task-spec/ops/gather.h"
#include "kernels/gather_kernels.h"
#include "op-attrs/ff_ordered/get_idxs.h"
#include "task-spec/profiling.h"
#include "utils/nonnegative_int/nonnegative_range.h"
#include <optional>

namespace FlexFlow {

using namespace FlexFlow::Kernels::Gather;

enum Slots {
  INPUT,
  OUTPUT,
  INDEX,
  ATTRS,
  HANDLE,
  PROFILING,
  PER_DEVICE_STATE,
  KERNEL_DEVICE_TYPE
};

OpTaskInvocation init(GatherAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind(INPUT, input_tensor(0_n));
  binding.bind(INDEX, input_tensor(1_n));
  binding.bind(OUTPUT, output_tensor(0_n));
  binding.bind_arg(ATTRS, attrs);
  binding.bind_arg(HANDLE, ff_handle());
  binding.bind_arg(KERNEL_DEVICE_TYPE, kernel_device_type());

  return OpTaskInvocation{
      task_id_t::GATHER_INIT_TASK_ID,
      binding,
  };
}

OpTaskInvocation forward(GatherAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(ATTRS, attrs);
  binding.bind_arg(PROFILING, profiling_settings());
  binding.bind_arg(KERNEL_DEVICE_TYPE, kernel_device_type());
  binding.bind_arg(PER_DEVICE_STATE,
                   per_device_op_state<std::optional<GatherPerDeviceState>>());

  binding.bind(INPUT, input_tensor(0_n));
  binding.bind(OUTPUT, output_tensor(0_n));
  binding.bind(INDEX, weight_tensor(0_n));

  return OpTaskInvocation{
      task_id_t::GATHER_FWD_TASK_ID,
      binding,
  };
}

OpTaskInvocation backward(GatherAttrs const &attrs) {
  OpTaskBinding binding = infer_bwd_binding(forward(attrs).binding);

  return OpTaskInvocation{
      task_id_t::GATHER_BWD_TASK_ID,
      binding,
  };
}

static DeviceSpecificPerDeviceOpState
    init_task_impl(TaskArgumentAccessor const &acc) {
  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto index = acc.get_tensor<Permissions::RO>(INDEX);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

  device_handle_t handle = acc.get_argument<device_handle_t>(HANDLE);
  DeviceType kernel_device_type =
      acc.get_argument<DeviceType>(KERNEL_DEVICE_TYPE);
  auto const &attrs = acc.get_argument<GatherAttrs>(ATTRS);

  ASSERT(get_num_dims(input.shape.dims) == get_num_dims(index.shape.dims));
  ASSERT(get_num_dims(output.shape.dims) == get_num_dims(index.shape.dims));

  for (ff_dim_t i : get_idxs(input.shape.dims.ff_ordered)) {
    ASSERT(dim_at_idx(index.shape.dims, i) == dim_at_idx(output.shape.dims, i));
    if (i != attrs.dim) {
      ASSERT(dim_at_idx(input.shape.dims, i) ==
             dim_at_idx(index.shape.dims, i));
    }
  }

  std::optional<GatherPerDeviceState> per_device_state =
      init_kernel(kernel_device_type, handle, attrs.dim);
  return DeviceSpecificPerDeviceOpState{
      DeviceSpecific<std::optional<GatherPerDeviceState>>::create(
          per_device_state),
  };
}

static std::optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  DeviceType kernel_device_type =
      acc.get_argument<DeviceType>(KERNEL_DEVICE_TYPE);
  auto per_device_state =
      acc.get_argument<GatherPerDeviceState>(PER_DEVICE_STATE);

  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto index = acc.get_tensor<Permissions::RO>(INDEX);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

  return profile(forward_kernel,
                 profiling,
                 kernel_device_type,
                 "[Gather] forward_time = {:.2lf}ms\n",
                 per_device_state,
                 input,
                 index,
                 output);
}

static std::optional<float>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  DeviceType kernel_device_type =
      acc.get_argument<DeviceType>(KERNEL_DEVICE_TYPE);
  auto per_device_state =
      acc.get_argument<GatherPerDeviceState>(PER_DEVICE_STATE);

  auto output_grad = acc.get_tensor_grad<Permissions::RO>(OUTPUT);
  auto index = acc.get_tensor<Permissions::RO>(INDEX);
  auto input_grad = acc.get_tensor_grad<Permissions::WO>(INPUT);

  return profile(backward_kernel,
                 profiling,
                 kernel_device_type,
                 "[Gather] backward_time = {:.2lf}ms\n",
                 per_device_state,
                 output_grad,
                 index,
                 input_grad);
}

TaskImplFunction get_gather_init_task_impl() {
  return TaskImplFunction{InitOpTaskImplFunction{init_task_impl}};
}
TaskImplFunction get_gather_fwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{forward_task_impl}};
}
TaskImplFunction get_gather_bwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{backward_task_impl}};
}

OpTaskSignature get_gather_init_signature() {
  OpTaskSignature init(OpTaskType::INIT);

  init.add_input_slot(INPUT);
  init.add_input_slot(INDEX);
  init.add_output_slot(OUTPUT);

  init.add_arg_slot<GatherAttrs>(ATTRS);
  init.add_arg_slot<DeviceType>(KERNEL_DEVICE_TYPE);
  init.add_unchecked_arg_slot<device_handle_t>(HANDLE);

  init.add_return_value<GatherPerDeviceState>();

  return init;
}

OpTaskSignature get_gather_fwd_signature() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_arg_slot<bool>(PROFILING);
  fwd.add_arg_slot<DeviceType>(KERNEL_DEVICE_TYPE);
  fwd.add_arg_slot<GatherAttrs>(ATTRS);

  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT);
  fwd.add_weight_slot(INDEX);

  return fwd;
}

OpTaskSignature get_gather_bwd_signature() {
  OpTaskSignature bwd = infer_bwd_signature(get_gather_fwd_signature());

  return bwd;
}

std::vector<task_id_t> get_task_ids(GatherAttrs const &) {
  return {task_id_t::GATHER_INIT_TASK_ID,
          task_id_t::GATHER_FWD_TASK_ID,
          task_id_t::GATHER_BWD_TASK_ID};
}

}; // namespace FlexFlow
