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

#include "task-spec/ops/layer_norm.h"
#include "kernels/layer_norm_kernels.h"
#include "op-attrs/ops/layer_norm.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "task-spec/profiling.h"
#include "utils/containers/product.h"
#include "utils/exception.h"
#include "utils/hash-utils.h"
#include "utils/nonnegative_int/nonnegative_range.h"
#include <type_traits>
#include "op-attrs/ff_ordered/transform.h"

namespace FlexFlow {

using namespace FlexFlow::Kernels::LayerNorm;

enum Slots {
  PROFILING,
  INPUT,
  OUTPUT,
  GAMMA,
  BETA,
  PER_DEVICE_STATE,
  ATTRS,
  HANDLE,
  KERNEL_DEVICE_TYPE,
};

OpTaskInvocation init(LayerNormAttrs const &attrs) {
  OpTaskBinding b;

  b.bind(INPUT, input_tensor(0_n));

  b.bind_arg(HANDLE, ff_handle());
  b.bind_arg(KERNEL_DEVICE_TYPE, kernel_device_type());
  b.bind_arg(ATTRS, attrs);

  return OpTaskInvocation{
      task_id_t::LAYERNORM_INIT_TASK_ID,
      b,
  };
}

OpTaskInvocation forward(LayerNormAttrs const &attrs) {
  OpTaskBinding b;

  b.bind(INPUT, input_tensor(0_n));
  b.bind(OUTPUT, output_tensor(0_n));
  b.bind(GAMMA, weight_tensor(0_n));
  b.bind(BETA, weight_tensor(1_n));
  b.bind_arg(PROFILING, profiling_settings());
  b.bind_arg(KERNEL_DEVICE_TYPE, kernel_device_type());
  b.bind_arg(PER_DEVICE_STATE,
             per_device_op_state<std::optional<LayerNormPerDeviceState>>());

  return OpTaskInvocation{
      task_id_t::LAYERNORM_FWD_TASK_ID,
      b,
  };
}

OpTaskInvocation backward(LayerNormAttrs const &attrs) {
  OpTaskBinding b = infer_bwd_binding(forward(attrs).binding);

  return OpTaskInvocation{
      task_id_t::LAYERNORM_BWD_TASK_ID,
      b,
  };
}

static std::optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);
  auto gamma = acc.get_tensor<Permissions::RW>(GAMMA);
  auto beta = acc.get_tensor<Permissions::RW>(BETA);

  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  DeviceType kernel_device_type =
      acc.get_argument<DeviceType>(KERNEL_DEVICE_TYPE);
  auto &state = acc.get_argument<LayerNormPerDeviceState>(PER_DEVICE_STATE);

  return profile(forward_kernel,
                 profiling,
                 kernel_device_type,
                 "[LayerNorm] forward time = {:.2lf}ms\n",
                 state,
                 input,
                 output,
                 gamma,
                 beta);
}

static std::optional<float>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto gamma = acc.get_tensor<Permissions::RO>(GAMMA);

  auto input_grad = acc.get_tensor_grad<Permissions::RW>(INPUT);
  auto gamma_grad = acc.get_tensor_grad<Permissions::RW>(GAMMA);
  auto beta_grad = acc.get_tensor_grad<Permissions::RW>(BETA);
  auto output_grad = acc.get_tensor_grad<Permissions::RO>(OUTPUT);

  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  DeviceType kernel_device_type =
      acc.get_argument<DeviceType>(KERNEL_DEVICE_TYPE);
  auto &state = acc.get_argument<LayerNormPerDeviceState>(PER_DEVICE_STATE);

  return profile(backward_kernel,
                 profiling,
                 kernel_device_type,
                 "[LayerNorm] backward time = {:.2lf}ms\n",
                 state,
                 output_grad,
                 input,
                 input_grad,
                 gamma,
                 gamma_grad,
                 beta_grad);
}

static DeviceSpecificDeviceStates
    init_task_impl(TaskArgumentAccessor const &acc) {
  auto const &attrs = acc.get_argument<LayerNormAttrs>(ATTRS);
  DeviceType kernel_device_type =
      acc.get_argument<DeviceType>(KERNEL_DEVICE_TYPE);
  Allocator allocator = acc.get_allocator();
  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto handle = acc.get_argument<device_handle_t>(HANDLE);

  positive_int M = product(transform(attrs.axes, 
                                     [&](ff_dim_t dim) {
                                       return dim_at_idx(input.shape.dims, dim);
                                     }));

  positive_int num_replicas = get_num_elements(input.shape.dims);

  positive_int effective_num_elements = M;
  positive_int effective_batch_size =
      positive_int{get_num_elements(input.shape.dims) / M};

  std::optional<LayerNormPerDeviceState> per_device_state =
      init_kernel(kernel_device_type,
                  handle,
                  allocator,
                  attrs.elementwise_affine,
                  effective_batch_size.int_from_positive_int(),
                  effective_num_elements.int_from_positive_int(),
                  attrs.eps);

  return DeviceSpecificDeviceStates{
      DeviceSpecific<std::optional<LayerNormPerDeviceState>>::create(
          per_device_state),
  };
}

TaskImplFunction get_layer_norm_init_task_impl() {
  return TaskImplFunction{InitOpTaskImplFunction{init_task_impl}};
}
TaskImplFunction get_layer_norm_fwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{forward_task_impl}};
}
TaskImplFunction get_layer_norm_bwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{backward_task_impl}};
}

OpTaskSignature get_layer_norm_fwd_signature() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT);
  fwd.add_weight_slot(GAMMA);
  fwd.add_weight_slot(BETA);

  fwd.add_arg_slot<ProfilingSettings>(PROFILING);
  fwd.add_arg_slot<DeviceType>(KERNEL_DEVICE_TYPE);
  fwd.add_unchecked_arg_slot<LayerNormPerDeviceState>(PER_DEVICE_STATE);
  return fwd;
}

OpTaskSignature get_layer_norm_bwd_signature() {
  OpTaskSignature bwd = infer_bwd_signature(get_layer_norm_fwd_signature());
  return bwd;
}

OpTaskSignature get_layer_norm_init_signature() {
  OpTaskSignature init(OpTaskType::INIT);

  init.add_input_slot(INPUT);
  init.add_arg_slot<LayerNormAttrs>(ATTRS);
  init.add_arg_slot<DeviceType>(KERNEL_DEVICE_TYPE);
  init.add_unchecked_arg_slot<device_handle_t>(HANDLE);

  init.add_return_value<LayerNormPerDeviceState>();
  return init;
}

std::vector<task_id_t> get_task_ids(LayerNormAttrs const &) {
  return {task_id_t::LAYERNORM_INIT_TASK_ID,
          task_id_t::LAYERNORM_FWD_TASK_ID,
          task_id_t::LAYERNORM_BWD_TASK_ID};
}

} // namespace FlexFlow
