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

#include "task-spec/ops/repartition.h"
#include "kernels/partition_kernels.h"
#include "task-spec/profiling.h"
#include "utils/exception.h"
#include "utils/hash-utils.h"

namespace FlexFlow {

using namespace FlexFlow::Kernels::Repartition;

enum Slots {
  INPUT,
  OUTPUT,
  ATTRS,
  PROFILING,
  HANDLE,
  PER_DEVICE_STATE,
  KERNEL_DEVICE_TYPE
};

OpTaskInvocation init(RepartitionAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(HANDLE, ff_handle());
  binding.bind_arg(KERNEL_DEVICE_TYPE, kernel_device_type());

  binding.bind(INPUT, input_tensor(0_n));

  return OpTaskInvocation{
      task_id_t::REPARTITION_INIT_TASK_ID,
      binding,
  };
}

OpTaskInvocation forward(RepartitionAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(PROFILING, profiling_settings());
  binding.bind_arg(ATTRS, attrs);
  binding.bind_arg(KERNEL_DEVICE_TYPE, kernel_device_type());
  binding.bind_arg(
      PER_DEVICE_STATE,
      per_device_op_state<std::optional<RepartitionPerDeviceState>>());

  binding.bind(INPUT, input_tensor(0_n));
  binding.bind(OUTPUT, output_tensor(0_n));

  return OpTaskInvocation{
      task_id_t::REPARTITION_FWD_TASK_ID,
      binding,
  };
}

OpTaskInvocation backward(RepartitionAttrs const &attrs) {
  OpTaskBinding binding = infer_bwd_binding(forward(attrs).binding);

  return OpTaskInvocation{
      task_id_t::REPARTITION_BWD_TASK_ID,
      binding,
  };
}

static DeviceSpecificDeviceStates
    init_task_impl(TaskArgumentAccessor const &acc) {
  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  PerDeviceFFHandle handle = acc.get_argument<PerDeviceFFHandle>(HANDLE);
  DeviceType kernel_device_type =
      acc.get_argument<DeviceType>(KERNEL_DEVICE_TYPE);

  std::optional<RepartitionPerDeviceState> per_device_state =
      init_kernel(kernel_device_type, handle, input.data_type);

  return DeviceSpecificDeviceStates{
      DeviceSpecific<std::optional<RepartitionPerDeviceState>>::create(
          per_device_state),
  };
}

static std::optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  DeviceType kernel_device_type =
      acc.get_argument<DeviceType>(KERNEL_DEVICE_TYPE);
  auto per_device_state =
      acc.get_argument<RepartitionPerDeviceState>(PER_DEVICE_STATE);
  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

  return profile(forward_kernel,
                 profiling,
                 kernel_device_type,
                 "[Reparition/Partition] forward_time = {:.2lf}ms\n",
                 per_device_state,
                 input,
                 output);
}

static std::optional<float>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  DeviceType kernel_device_type =
      acc.get_argument<DeviceType>(KERNEL_DEVICE_TYPE);
  auto per_device_state =
      acc.get_argument<RepartitionPerDeviceState>(PER_DEVICE_STATE);

  auto output_grad = acc.get_tensor_grad<Permissions::RO>(INPUT);
  auto input_grad = acc.get_tensor_grad<Permissions::WO>(OUTPUT);

  return profile(backward_kernel,
                 profiling,
                 kernel_device_type,
                 "[Reparition/Partition] backward_time = {:.2lf}ms\n",
                 per_device_state,
                 output_grad,
                 input_grad);
}

TaskImplFunction get_repartition_init_task_impl() {
  return TaskImplFunction{InitOpTaskImplFunction{init_task_impl}};
}

TaskImplFunction get_repartition_fwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{forward_task_impl}};
}

TaskImplFunction get_repartition_bwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{backward_task_impl}};
}

OpTaskSignature get_repartition_init_signature() {
  OpTaskSignature init(OpTaskType::INIT);

  init.add_unchecked_arg_slot<PerDeviceFFHandle>(HANDLE);
  init.add_arg_slot<DeviceType>(KERNEL_DEVICE_TYPE);
  init.add_input_slot(INPUT);
  init.add_return_value<RepartitionPerDeviceState>();
  return init;
}

OpTaskSignature get_repartition_fwd_signature() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT);
  fwd.add_arg_slot<ProfilingSettings>(PROFILING);
  fwd.add_arg_slot<DeviceType>(KERNEL_DEVICE_TYPE);
  fwd.add_unchecked_arg_slot<RepartitionPerDeviceState>(PER_DEVICE_STATE);
  return fwd;
}

OpTaskSignature get_repartition_bwd_signature() {
  OpTaskSignature bwd = infer_bwd_signature(get_repartition_fwd_signature());
  return bwd;
}

std::vector<task_id_t> get_task_ids(RepartitionAttrs const &) {
  return {task_id_t::REPARTITION_INIT_TASK_ID,
          task_id_t::REPARTITION_FWD_TASK_ID,
          task_id_t::REPARTITION_BWD_TASK_ID};
}

}; // namespace FlexFlow
