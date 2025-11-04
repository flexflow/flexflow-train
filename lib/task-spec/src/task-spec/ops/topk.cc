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

#include "task-spec/ops/topk.h"
#include "kernels/topk_kernels.h"
#include "task-spec/profiling.h"
#include "utils/exception.h"

namespace FlexFlow {

using namespace FlexFlow::Kernels::TopK;

// For an input tensor, computes the top k entries in each row
// (resp. vector along the last dimension). Thus,
// values.shape = indices.shape = input.shape[:-1] + [k]

enum Slots { INPUT, OUTPUT, INDICES, ATTRS, PROFILING, KERNEL_DEVICE_TYPE };

OpTaskInvocation forward(TopKAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(PROFILING, profiling_settings());
  binding.bind_arg(ATTRS, attrs);
  binding.bind_arg(KERNEL_DEVICE_TYPE, kernel_device_type());

  binding.bind(INPUT, input_tensor(0_n));
  binding.bind(OUTPUT, output_tensor(0_n));
  binding.bind(INDICES, output_tensor(1_n));

  return OpTaskInvocation{
      op_task_id_t::FWD,
      binding,
  };
}

OpTaskInvocation backward(TopKAttrs const &attrs) {
  OpTaskBinding binding = infer_bwd_binding(forward(attrs).binding);

  return OpTaskInvocation{
      op_task_id_t::BWD,
      binding,
  };
}

static std::optional<milliseconds_t> forward_task_impl(TaskArgumentAccessor const &acc) {
  auto attrs = acc.get_argument<TopKAttrs>(ATTRS);
  auto profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  DeviceType kernel_device_type =
      acc.get_argument<DeviceType>(KERNEL_DEVICE_TYPE);

  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

  positive_int length = dim_at_idx(input.shape.dims, legion_dim_t{0_n});
  positive_int batch_size =
      positive_int{get_num_elements(input.shape.dims) / length};
  auto indices = acc.get_tensor<Permissions::WO>(INDICES);

  return profile(forward_kernel,
                 profiling,
                 kernel_device_type,
                 "[TopK] forward_time = {:.2lf}ms\n",
                 input.get_float_ptr(),
                 output.get_float_ptr(),
                 indices.get_int32_ptr(),
                 batch_size.int_from_positive_int(),
                 length.int_from_positive_int(),
                 attrs.k.int_from_positive_int(),
                 attrs.sorted);
}

static std::optional<milliseconds_t>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  auto attrs = acc.get_argument<TopKAttrs>(ATTRS);
  auto profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  DeviceType kernel_device_type =
      acc.get_argument<DeviceType>(KERNEL_DEVICE_TYPE);

  auto input_grad = acc.get_tensor_grad<Permissions::RW>(INPUT);
  auto output_grad = acc.get_tensor_grad<Permissions::RO>(OUTPUT);

  auto indices = acc.get_tensor<Permissions::RO>(INDICES);

  positive_int length = dim_at_idx(input_grad.shape.dims, legion_dim_t{0_n});
  positive_int batch_size =
      positive_int{get_num_elements(input_grad.shape.dims) / length};

  return profile(backward_kernel,
                 profiling,
                 kernel_device_type,
                 "[TopK] backward_time = {:.2lf}ms\n",
                 output_grad.get_float_ptr(),
                 indices.get_int32_ptr(),
                 input_grad.get_float_ptr(),
                 batch_size.int_from_positive_int(),
                 length.int_from_positive_int(),
                 attrs.k.int_from_positive_int());
}

TaskImplFunction get_topk_fwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{forward_task_impl}};
}

TaskImplFunction get_topk_bwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{backward_task_impl}};
}

OpTaskSignature get_topk_fwd_signature() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_arg_slot<ProfilingSettings>(PROFILING);
  fwd.add_arg_slot<TopKAttrs>(ATTRS);
  fwd.add_arg_slot<DeviceType>(KERNEL_DEVICE_TYPE);

  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT);
  fwd.add_output_slot(INDICES);
  return fwd;
}

OpTaskSignature get_topk_bwd_signature() {
  OpTaskSignature bwd = infer_bwd_signature(get_topk_fwd_signature());
  return bwd;
}

}; // namespace FlexFlow
