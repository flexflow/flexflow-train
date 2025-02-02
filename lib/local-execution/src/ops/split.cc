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

#include "split.h"
#include "kernels/array_shape.h"
#include "kernels/split_kernels.h"
#include "op-attrs/get_output_shapes.h"
#include "utils/exception.h"
#include "utils/hash-utils.h"
#include "utils/nonnegative_int/nonnegative_range.h"

namespace FlexFlow {

using namespace FlexFlow::Kernels::Split;
using coord_t = long long;

enum Slots { INPUT, OUTPUT, ATTRS, PROFILING };

OpTaskInvocation forward(SplitAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(PROFILING, profiling_settings());
  binding.bind_arg(ATTRS, attrs);
  binding.bind(INPUT, input_tensor(0));
  binding.bind(OUTPUT, output_tensor(0));

  return {task_id_t::SPLIT_FWD_TASK_ID, binding};
}

OpTaskInvocation backward(SplitAttrs const &attrs) {
  OpTaskBinding binding = infer_bwd_binding(forward(attrs).binding);

  return {task_id_t::SPLIT_BWD_TASK_ID, binding};
}

static std::pair<nonnegative_int, nonnegative_int>
    calc_block_size(ArrayShape const &array_shape, ff_dim_t axis) {
  nonnegative_int num_blocks = 1_n;
  nonnegative_int block_size = 1_n;
  for (nonnegative_int d : nonnegative_range(array_shape.num_elements())) {
    if (d <= axis.value) {
      block_size *= array_shape.at(legion_dim_t{d});
    } else {
      num_blocks *= array_shape.at(legion_dim_t{d});
    }
  }
  return {num_blocks, block_size};
}

static std::optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);
  auto attrs = acc.get_argument<SplitAttrs>(ATTRS);

  coord_t out_block_sizes[MAX_NUM_OUTPUTS];
  auto [num_blocks, in_block_size] = calc_block_size(input.shape, attrs.axis);

  for (int i = 0; i < attrs.splits.size(); i++) {
    auto [_, out_block_size] = calc_block_size(output.shape, attrs.axis);
    out_block_sizes[i] = out_block_size.unwrap_nonnegative();
  }
  float *output_float_ptr = output.get_float_ptr();
  return profile(forward_kernel,
                 profiling,
                 "Split forward_time = {:.2lf}ms\n",
                 &output_float_ptr,
                 input.get_float_ptr(),
                 out_block_sizes,
                 in_block_size.unwrap_nonnegative(),
                 num_blocks.unwrap_nonnegative(),
                 attrs.splits.size());
}

// maybe we should add assert like the original code
static std::optional<float>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  auto input_grad = acc.get_tensor_grad<Permissions::RW>(INPUT);
  auto output_grad = acc.get_tensor_grad<Permissions::RO>(OUTPUT);
  auto attrs = acc.get_argument<SplitAttrs>(ATTRS);

  coord_t out_block_sizes[MAX_NUM_OUTPUTS];
  auto [num_blocks, in_block_size] =
      calc_block_size(input_grad.shape, attrs.axis);

  for (int i = 0; i < attrs.splits.size(); i++) {
    coord_t out_num_blocks;
    auto [_, out_block_size] = calc_block_size(output_grad.shape, attrs.axis);
    out_block_sizes[i] = out_block_size.unwrap_nonnegative();
  }
  float const *output_grad_ptr = output_grad.get_float_ptr();
  return profile(backward_kernel,
                 profiling,
                 "Split backward_time = {:.2lf}ms\n",
                 input_grad.get_float_ptr(),
                 &output_grad_ptr,
                 out_block_sizes,
                 in_block_size.unwrap_nonnegative(),
                 num_blocks.unwrap_nonnegative(),
                 attrs.splits.size());
}

TaskImplFunction get_split_fwd_task_impl() {
  return TaskImplFunction{FwdBwdTaskImplFunction{forward_task_impl}};
}
TaskImplFunction get_split_bwd_task_impl() {
  return TaskImplFunction{FwdBwdTaskImplFunction{backward_task_impl}};
}

OpTaskSignature get_split_fwd_signature() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_arg_slot<ProfilingSettings>(PROFILING);

  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT);
  return fwd;
}
OpTaskSignature get_split_bwd_signature() {
  OpTaskSignature bwd = infer_bwd_signature(get_split_fwd_signature());
  return bwd;
}

std::vector<task_id_t> get_task_ids(SplitAttrs const &) {
  return {task_id_t::SPLIT_FWD_TASK_ID, task_id_t::SPLIT_BWD_TASK_ID};
}

}; // namespace FlexFlow
