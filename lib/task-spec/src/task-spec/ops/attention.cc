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

#include "task-spec/ops/attention.h"
#include "kernels/attention_kernels.h"
#include "kernels/device_handle_t.dtg.h"
#include "op-attrs/ops/attention.h"
#include "op-attrs/ops/attention/multihead_attention_parallel_inputs.h"
#include "task-spec/op_task_signature.h"
#include "task-spec/profiling.h"

namespace FlexFlow {

using namespace FlexFlow::Kernels::MultiHeadAttention;

enum Slots {
  QUERY_PARALLEL_TENSOR_SHAPE,
  KEY_PARALLEL_TENSOR_SHAPE,
  VALUE_PARALLEL_TENSOR_SHAPE,
  QPROJSIZE,
  KPROJSIZE,
  VPROJSIZE,
  OPROJSIZE,
  ATTRS,
  PROFILING,
  QUERY,
  KEY,
  VALUE,
  WEIGHTS,
  OUTPUT,
  HANDLE,
  PER_DEVICE_STATE,
  KERNEL_DEVICE_TYPE,
};

OpTaskInvocation init(MultiHeadAttentionAttrs const &attrs) {
  OpTaskBinding b;

  b.bind_arg(HANDLE, ff_handle());
  b.bind_arg(ATTRS, attrs);

  b.bind_arg(KERNEL_DEVICE_TYPE, kernel_device_type());

  b.bind_arg(QUERY_PARALLEL_TENSOR_SHAPE, input_parallel_tensor_shape(0_n));
  b.bind_arg(KEY_PARALLEL_TENSOR_SHAPE, input_parallel_tensor_shape(1_n));
  b.bind_arg(VALUE_PARALLEL_TENSOR_SHAPE, input_parallel_tensor_shape(2_n));

  b.bind_arg(QPROJSIZE, get_qProjSize(attrs));
  b.bind_arg(KPROJSIZE, get_kProjSize(attrs));
  b.bind_arg(VPROJSIZE, get_vProjSize(attrs));
  b.bind_arg(OPROJSIZE, get_oProjSize(attrs));

  return OpTaskInvocation{
      task_id_t::ATTENTION_INIT_TASK_ID,
      b,
  };
}

OpTaskInvocation forward(MultiHeadAttentionAttrs const &attrs) {
  OpTaskBinding b;

  b.bind(QUERY, input_tensor(0_n));
  b.bind(KEY, input_tensor(1_n));
  b.bind(VALUE, input_tensor(2_n));
  b.bind(WEIGHTS, weight_tensor(0_n));
  b.bind(OUTPUT, output_tensor(0_n));

  b.bind_arg(KERNEL_DEVICE_TYPE, kernel_device_type());
  b.bind_arg(PROFILING, profiling_settings());
  b.bind_arg(PER_DEVICE_STATE,
             per_device_op_state<std::optional<MHAPerDeviceState>>());

  return OpTaskInvocation{
      task_id_t::ATTENTION_FWD_TASK_ID,
      b,
  };
}

OpTaskInvocation backward(MultiHeadAttentionAttrs const &attrs) {
  OpTaskBinding b = infer_bwd_binding(forward(attrs).binding);

  return OpTaskInvocation{
      task_id_t::ATTENTION_BWD_TASK_ID,
      b,
  };
}

static DeviceSpecificPerDeviceOpState
    init_task_impl(TaskArgumentAccessor const &acc) {
  auto const &attrs = acc.get_argument<MultiHeadAttentionAttrs>(ATTRS);
  Allocator allocator = acc.get_allocator();

  DeviceType kernel_device_type =
      acc.get_argument<DeviceType>(KERNEL_DEVICE_TYPE);

  positive_int qProjSize = acc.get_argument<positive_int>(QPROJSIZE);
  positive_int kProjSize = acc.get_argument<positive_int>(KPROJSIZE);
  positive_int vProjSize = acc.get_argument<positive_int>(VPROJSIZE);
  positive_int oProjSize = acc.get_argument<positive_int>(OPROJSIZE);

  device_handle_t handle = acc.get_argument<device_handle_t>(HANDLE);
  ParallelTensorShape query_parallel_tensor_shape =
      acc.get_argument<ParallelTensorShape>(QUERY_PARALLEL_TENSOR_SHAPE);
  ParallelTensorShape key_parallel_tensor_shape =
      acc.get_argument<ParallelTensorShape>(KEY_PARALLEL_TENSOR_SHAPE);
  ParallelTensorShape value_parallel_tensor_shape =
      acc.get_argument<ParallelTensorShape>(VALUE_PARALLEL_TENSOR_SHAPE);

  MultiHeadAttentionParallelInputs parsed = throw_if_unexpected(
      parse_attention_parallel_input_shape(query_parallel_tensor_shape,
                                           key_parallel_tensor_shape,
                                           value_parallel_tensor_shape));
  ParallelTensorShape weight_parallel_tensor_shape =
      throw_if_unexpected(get_weights_shape(attrs,
                                            query_parallel_tensor_shape,
                                            key_parallel_tensor_shape,
                                            value_parallel_tensor_shape));

  positive_int kvSeqLength = get_kvSeqLength(parsed);
  positive_int qSize = get_qSize(parsed);
  positive_int kSize = get_kSize(parsed);
  positive_int vSize = get_vSize(parsed);

  positive_int qoSeqLength = get_qoSeqLength(parsed);
  positive_int num_samples = get_num_samples(parsed);
  positive_int num_heads = attrs.num_heads;

  std::optional<MHAPerDeviceState> per_device_state = init_kernel(
      /*device_type=*/kernel_device_type,
      /*per_device_ff_handle=*/handle,
      /*allocator=*/allocator,
      /*num_samples=*/num_samples.int_from_positive_int(),
      /*num_heads=*/num_heads.int_from_positive_int(),
      /*qSize=*/qSize.int_from_positive_int(),
      /*kSize=*/kSize.int_from_positive_int(),
      /*vSize=*/vSize.int_from_positive_int(),
      /*qProjSize=*/qProjSize.int_from_positive_int(),
      /*kProjSize=*/kProjSize.int_from_positive_int(),
      /*vProjSize=*/vProjSize.int_from_positive_int(),
      /*oProjSize=*/oProjSize.int_from_positive_int(),
      /*qoSeqLength=*/qoSeqLength.int_from_positive_int(),
      /*kvSeqLength=*/kvSeqLength.int_from_positive_int(),
      /*add_bias_kv=*/attrs.add_bias_kv);

  return DeviceSpecificPerDeviceOpState{
    acc.make_device_specific(per_device_state),
  };
}

static std::optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  auto query = acc.get_tensor<Permissions::RO>(QUERY);
  auto key = acc.get_tensor<Permissions::RO>(KEY);
  auto value = acc.get_tensor<Permissions::RO>(VALUE);
  auto weight = acc.get_tensor<Permissions::RO>(WEIGHTS);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  DeviceType kernel_device_type =
      acc.get_argument<DeviceType>(KERNEL_DEVICE_TYPE);
  std::optional<MHAPerDeviceState> per_device_state =
      acc.get_argument<std::optional<MHAPerDeviceState>>(PER_DEVICE_STATE);

  return profile(forward_kernel,
                 profiling,
                 kernel_device_type,
                 "[MultiHeadAttention] forward_time = {:.2lf}ms\n",
                 per_device_state,
                 query.get_float_ptr(),
                 key.get_float_ptr(),
                 value.get_float_ptr(),
                 weight.get_float_ptr(),
                 output.get_float_ptr());
}

static std::optional<float>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  auto query = acc.get_tensor<Permissions::RO>(QUERY);
  auto key = acc.get_tensor<Permissions::RO>(KEY);
  auto value = acc.get_tensor<Permissions::RO>(VALUE);
  auto weight = acc.get_tensor<Permissions::RO>(WEIGHTS);

  auto output_grad = acc.get_tensor_grad<Permissions::RO>(OUTPUT);
  auto weight_grad = acc.get_tensor_grad<Permissions::RW>(WEIGHTS);
  auto query_grad = acc.get_tensor_grad<Permissions::RW>(QUERY);
  auto key_grad = acc.get_tensor_grad<Permissions::RW>(KEY);
  auto value_grad = acc.get_tensor_grad<Permissions::RW>(VALUE);

  std::optional<MHAPerDeviceState> per_device_state =
      acc.get_argument<std::optional<MHAPerDeviceState>>(PER_DEVICE_STATE);
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  DeviceType kernel_device_type =
      acc.get_argument<DeviceType>(KERNEL_DEVICE_TYPE);

  float *key_grad_ptr =
      (key_grad == query_grad) ? nullptr : key_grad.get_float_ptr();
  float *value_grad_ptr = (value_grad == query_grad || value_grad == key_grad)
                              ? nullptr
                              : value_grad.get_float_ptr();

  ASSERT(value_grad.shape == value.shape);
  ASSERT(key_grad.shape == key.shape);

  ASSERT(query_grad.shape == query.shape);
  ASSERT(weight_grad.shape == weight.shape);

  return profile(backward_kernel,
                 profiling,
                 kernel_device_type,
                 "[MultiHeadAttention] backward_time = {:.2lf}ms\n",
                 per_device_state,
                 query.get_float_ptr(),
                 query_grad.get_float_ptr(),
                 key.get_float_ptr(),
                 key_grad_ptr,
                 value.get_float_ptr(),
                 value_grad_ptr,
                 weight.get_float_ptr(),
                 weight_grad.get_float_ptr(),
                 output_grad.get_float_ptr());
}

TaskImplFunction get_attention_init_task_impl() {
  return TaskImplFunction{InitOpTaskImplFunction{init_task_impl}};
}
TaskImplFunction get_attention_fwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{forward_task_impl}};
}
TaskImplFunction get_attention_bwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{backward_task_impl}};
}

OpTaskSignature get_attention_init_signature() {
  OpTaskSignature init(OpTaskType::INIT);
  init.add_arg_slot<ParallelTensorShape>(QUERY_PARALLEL_TENSOR_SHAPE);
  init.add_arg_slot<ParallelTensorShape>(KEY_PARALLEL_TENSOR_SHAPE);
  init.add_arg_slot<ParallelTensorShape>(VALUE_PARALLEL_TENSOR_SHAPE);
  init.add_arg_slot<positive_int>(QPROJSIZE);
  init.add_arg_slot<positive_int>(KPROJSIZE);
  init.add_arg_slot<positive_int>(VPROJSIZE);
  init.add_arg_slot<positive_int>(OPROJSIZE);
  init.add_arg_slot<MultiHeadAttentionAttrs>(ATTRS);
  init.add_unchecked_arg_slot<PerDeviceFFHandle>(HANDLE);

  init.add_return_value<std::optional<MHAPerDeviceState>>();

  return init;
}

OpTaskSignature get_attention_fwd_signature() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_input_slot(QUERY);
  fwd.add_input_slot(KEY);
  fwd.add_input_slot(VALUE);
  fwd.add_weight_slot(WEIGHTS);
  fwd.add_output_slot(OUTPUT);

  fwd.add_arg_slot<ProfilingSettings>(PROFILING);
  fwd.add_unchecked_arg_slot<std::optional<MHAPerDeviceState>>(
      PER_DEVICE_STATE);

  return fwd;
}

OpTaskSignature get_attention_bwd_signature() {
  OpTaskSignature bwd = infer_bwd_signature(get_attention_fwd_signature());

  return bwd;
}

std::vector<task_id_t> get_task_ids(MultiHeadAttentionAttrs const &) {
  return {task_id_t::ATTENTION_INIT_TASK_ID,
          task_id_t::ATTENTION_FWD_TASK_ID,
          task_id_t::ATTENTION_BWD_TASK_ID};
}

} // namespace FlexFlow
