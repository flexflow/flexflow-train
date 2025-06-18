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

#include "task-spec/ops/batch_matmul.h"
#include "kernels/batch_matmul_kernels.h"
#include "op-attrs/ops/batch_matmul.h"
#include "task-spec/op_task_signature.h"
#include "utils/containers/transform.h"
#include "utils/nonnegative_int/nonnegative_range.h"

namespace FlexFlow {

using namespace FlexFlow::Kernels::BatchMatmul;

enum Slots {
  A_INPUT, // tensor
  B_INPUT, // tensor
  ATTRS,
  OUTPUT, // tensor
  PROFILING,
  HANDLE,
  ITERATION_CONFIG
};

OpTaskInvocation forward(BatchMatmulAttrs const &attrs) {
  OpTaskBinding fwd;

  fwd.bind(A_INPUT, input_tensor(0));
  fwd.bind(B_INPUT, input_tensor(1));
  fwd.bind(OUTPUT, output_tensor(0));

  fwd.bind_arg(ATTRS, attrs);
  fwd.bind_arg(HANDLE, ff_handle());
  fwd.bind_arg(PROFILING, profiling_settings());
  fwd.bind_arg(ITERATION_CONFIG, iteration_config());

  return {task_id_t::BATCHMATMUL_FWD_TASK_ID, fwd};
}

OpTaskInvocation backward(BatchMatmulAttrs const &attrs) {
  OpTaskBinding bwd = infer_bwd_binding(forward(attrs).binding);

  return {task_id_t::BATCHMATMUL_BWD_TASK_ID, bwd};
}

static std::optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  auto a_input = acc.get_tensor<Permissions::RO>(A_INPUT);
  auto b_input = acc.get_tensor<Permissions::RO>(B_INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);
  auto attrs = acc.get_argument<BatchMatmulAttrs>(ATTRS);
  PerDeviceFFHandle handle = acc.get_argument<PerDeviceFFHandle>(HANDLE);

  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  FFIterationConfig iter_config =
      acc.get_argument<FFIterationConfig>(ITERATION_CONFIG);

  positive_int m = b_input.shape.at(legion_dim_t{0_n});
  ASSERT(m == output.shape.at(legion_dim_t{0_n}));
  positive_int n = a_input.shape.at(legion_dim_t{1_n});
  ASSERT(n == output.shape.at(legion_dim_t{1_n}));
  positive_int k = a_input.shape.at(legion_dim_t{0_n});
  ASSERT(k == b_input.shape.at(legion_dim_t{1_n}));

  ASSERT(a_input.shape.num_elements() == b_input.shape.num_elements());
  ASSERT(a_input.shape.num_elements() == output.shape.num_elements());

  positive_int batch = 1_p;
  for (nonnegative_int i : nonnegative_range(2_n, a_input.shape.num_dims())) {
    positive_int dim_size = a_input.shape.at(legion_dim_t{i});
    ASSERT(dim_size == b_input.shape.at(legion_dim_t{i}));
    ASSERT(dim_size == output.shape.at(legion_dim_t{i}));
    batch *= dim_size;
  }

  auto get_raw_seq_len = [](std::optional<nonnegative_int> seq_len) -> int {
    return transform(seq_len,
                     [](nonnegative_int x) { return x.unwrap_nonnegative(); })
        .value_or(-1);
  };

  return profile(forward_kernel,
                 profiling,
                 "[BatchMatmul] forward_time = {:.2lf}ms\n",
                 handle,
                 output.get_float_ptr(),
                 a_input.get_float_ptr(),
                 b_input.get_float_ptr(),
                 m.int_from_positive_int(),
                 n.int_from_positive_int(),
                 k.int_from_positive_int(),
                 batch.int_from_positive_int(),
                 get_raw_seq_len(attrs.a_seq_length_dim),
                 get_raw_seq_len(attrs.b_seq_length_dim),
                 iter_config.seq_length);
}

static std::optional<float>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  // BatchMatmul* bmm = (BatchMatmul*) task->args;
  FFIterationConfig iter_config =
      acc.get_argument<FFIterationConfig>(ITERATION_CONFIG);
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  PerDeviceFFHandle handle = acc.get_argument<PerDeviceFFHandle>(HANDLE);

  auto output = acc.get_tensor<Permissions::RO>(OUTPUT);
  auto output_grad = acc.get_tensor_grad<Permissions::RW>(OUTPUT);
  ASSERT(output.shape == output_grad.shape);

  auto a_input = acc.get_tensor<Permissions::RO>(A_INPUT);
  auto a_input_grad = acc.get_tensor_grad<Permissions::RW>(A_INPUT);
  ASSERT(a_input.shape == a_input_grad.shape);

  auto b_input = acc.get_tensor<Permissions::RO>(B_INPUT);
  auto b_input_grad = acc.get_tensor_grad<Permissions::RW>(B_INPUT);
  ASSERT(b_input.shape == b_input_grad.shape);

  // check dins
  positive_int m = b_input.shape.at(legion_dim_t{0_n});
  ASSERT(m == output.shape.at(legion_dim_t{0_n}));
  positive_int n = a_input.shape.at(legion_dim_t{1_n});
  ASSERT(n == output.shape.at(legion_dim_t{1_n}));
  positive_int k = a_input.shape.at(legion_dim_t{0_n});
  ASSERT(k == b_input.shape.at(legion_dim_t{1_n}));
  ASSERT(a_input.shape.num_elements() == b_input.shape.num_elements());
  ASSERT(a_input.shape.num_elements() == output.shape.num_elements());

  positive_int batch = 1_p;
  for (nonnegative_int i : nonnegative_range(2_n, a_input.shape.num_dims())) {
    positive_int dim_size = a_input.shape.at(legion_dim_t{i});
    ASSERT(dim_size == b_input.shape.at(legion_dim_t{i}));
    ASSERT(dim_size == output.shape.at(legion_dim_t{i}));
    batch *= dim_size;
  }

  return profile(backward_kernel,
                 profiling,
                 "[BatchMatmul] backward_time = {:.2lf}ms\n",
                 handle,
                 output.get_float_ptr(),
                 output_grad.get_float_ptr(),
                 a_input.get_float_ptr(),
                 a_input_grad.get_float_ptr(),
                 b_input.get_float_ptr(),
                 b_input_grad.get_float_ptr(),
                 m.int_from_positive_int(),
                 n.int_from_positive_int(),
                 k.int_from_positive_int(),
                 batch.int_from_positive_int());
}

TaskImplFunction get_batch_matmul_fwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{forward_task_impl}};
}
TaskImplFunction get_batch_matmul_bwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{backward_task_impl}};
}

OpTaskSignature get_batch_matmul_fwd_signature() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_input_slot(A_INPUT);
  fwd.add_input_slot(B_INPUT);
  fwd.add_output_slot(OUTPUT);
  fwd.add_arg_slot<BatchMatmulAttrs>(ATTRS);
  fwd.add_arg_slot<ProfilingSettings>(PROFILING);
  fwd.add_unchecked_arg_slot<PerDeviceFFHandle>(HANDLE);

  return fwd;
}

OpTaskSignature get_batch_matmul_bwd_signature() {
  OpTaskSignature bwd = infer_bwd_signature(get_batch_matmul_fwd_signature());

  return bwd;
}

std::vector<task_id_t> get_task_ids(BatchMatmulAttrs const &) {
  return {task_id_t::BATCHMATMUL_FWD_TASK_ID,
          task_id_t::BATCHMATMUL_BWD_TASK_ID};
}

}; // namespace FlexFlow
