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

#include "task-spec/ops/impl/batch_matmul.h"
#include "kernels/batch_matmul_kernels.h"
#include "op-attrs/ops/batch_matmul.h"
#include "task-spec/profiling.h"
#include "utils/containers/transform.h"
#include "utils/nonnegative_int/nonnegative_range.h"

namespace FlexFlow {

using namespace FlexFlow::Kernels::BatchMatmul;

static std::optional<milliseconds_t> forward_task_impl(TaskArgumentAccessor const &acc) {
  auto a_input = acc.get_tensor<Permissions::RO>(TensorSlotName::LHS_INPUT);
  auto b_input = acc.get_tensor<Permissions::RO>(TensorSlotName::RHS_INPUT);
  auto output = acc.get_tensor<Permissions::WO>(TensorSlotName::OUTPUT);
  BatchMatmulAttrs attrs = acc.get_op_attrs().require_batch_matmul();
  device_handle_t handle = acc.get_ff_handle();

  ProfilingSettings profiling = acc.get_profiling_settings();
  FFIterationConfig iter_config = acc.get_iteration_config();
  DeviceType kernel_device_type = acc.get_kernel_device_type();

  return profile(forward_kernel,
                 profiling,
                 kernel_device_type,
                 "[BatchMatmul] forward_time = {:.2lf}ms\n",
                 handle,
                 output,
                 a_input,
                 b_input,
                 iter_config.seq_length,
                 attrs.a_seq_length_dim,
                 attrs.b_seq_length_dim);
}

static std::optional<milliseconds_t>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  FFIterationConfig iter_config = acc.get_iteration_config();
  ProfilingSettings profiling = acc.get_profiling_settings();
  device_handle_t handle = acc.get_ff_handle();
  DeviceType kernel_device_type = acc.get_kernel_device_type();

  auto output = acc.get_tensor<Permissions::RO>(TensorSlotName::OUTPUT);
  auto output_grad = acc.get_tensor_grad<Permissions::RW>(TensorSlotName::OUTPUT);
  ASSERT(output.shape == output_grad.shape);

  auto a_input = acc.get_tensor<Permissions::RO>(TensorSlotName::LHS_INPUT);
  auto a_input_grad = acc.get_tensor_grad<Permissions::RW>(TensorSlotName::LHS_INPUT);
  ASSERT(a_input.shape == a_input_grad.shape);

  auto b_input = acc.get_tensor<Permissions::RO>(TensorSlotName::RHS_INPUT);
  auto b_input_grad = acc.get_tensor_grad<Permissions::RW>(TensorSlotName::RHS_INPUT);
  ASSERT(b_input.shape == b_input_grad.shape);

  return profile(backward_kernel,
                 profiling,
                 kernel_device_type,
                 "[BatchMatmul] backward_time = {:.2lf}ms\n",
                 handle,
                 output,
                 output_grad,
                 a_input,
                 a_input_grad,
                 b_input,
                 b_input_grad);
}

TaskImplFunction get_batch_matmul_fwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{forward_task_impl}};
}

TaskImplFunction get_batch_matmul_bwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{backward_task_impl}};
}

}; // namespace FlexFlow
