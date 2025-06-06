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

#include "internal/device.h"
#include "kernels/accessor.h"
#include "kernels/legion_ordered/transform.h"
#include "kernels/transpose_kernels.h"
#include "utils/exception.h"
#include "utils/nonnegative_int/num_elements.h"

namespace FlexFlow {

struct TransposeStrides {
  int num_dim;
  int in_strides[MAX_TENSOR_DIM], out_strides[MAX_TENSOR_DIM],
      perm[MAX_TENSOR_DIM];
};

namespace Kernels {
namespace Transpose {

__global__ void transpose_simple_kernel(std::size_t volume,
                                        float const *in_ptr,
                                        float *out_ptr,
                                        const TransposeStrides info,
                                        float const beta) {
  CUDA_KERNEL_LOOP(o_idx, volume) {
    size_t i_idx = 0;
    size_t t = o_idx;
    for (int i = info.num_dim - 1; i >= 0; i--) {
      coord_t ratio = t / info.out_strides[i];
      t -= ratio * info.out_strides[i];
      i_idx += ratio * info.in_strides[info.perm[i]];
    }
    out_ptr[o_idx] += out_ptr[o_idx] * beta + in_ptr[i_idx];
  }
}

static LegionOrdered<legion_dim_t>
    legion_ordered_perm_from_ff_ordered(FFOrdered<ff_dim_t> const &perm) {
  nonnegative_int perm_size = num_elements(perm);
  LegionOrdered<legion_dim_t> legion_ordered_perm =
      transform(legion_ordered_from_ff_ordered(perm), [&](ff_dim_t d) {
        return legion_dim_from_ff_dim(d, perm_size);
      });

  return legion_ordered_perm;
}

void forward_kernel(cudaStream_t stream,
                    TransposeAttrs const &m,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output) {

  TransposeStrides info;
  info.num_dim = input.shape.num_dims().unwrap_nonnegative();
  assert(info.num_dim == m.perm.size());

  LegionOrdered<legion_dim_t> legion_ordered_perm =
      legion_ordered_perm_from_ff_ordered(m.perm);

  for (int i = 0; i < info.num_dim; i++) {
    if (i == 0) {
      info.in_strides[i] = 1;
      info.out_strides[i] = 1;
    } else {
      int in_dim_size =
          input.shape.at(legion_dim_t{nonnegative_int{i}}).unwrap_nonnegative();
      int out_dim_size = output.shape.at(legion_dim_t{nonnegative_int{i}})
                             .unwrap_nonnegative();
      info.in_strides[i] = info.in_strides[i - 1] * in_dim_size;
      info.out_strides[i] = info.out_strides[i - 1] * out_dim_size;
    }

    info.perm[i] = legion_ordered_perm.at(legion_dim_t{nonnegative_int{i}})
                       .value.unwrap_nonnegative();
  }
  transpose_simple_kernel<<<
      GET_BLOCKS(output.shape.get_volume().unwrap_nonnegative()),
      CUDA_NUM_THREADS,
      0,
      stream>>>(output.shape.get_volume().unwrap_nonnegative(),
                input.get_float_ptr(),
                output.get_float_ptr(),
                info,
                0.0f /*beta*/);
}

void backward_kernel(cudaStream_t stream,
                     TransposeAttrs const &m,
                     GenericTensorAccessorR const &out_grad,
                     GenericTensorAccessorW const &in_grad) {

  TransposeStrides info;
  info.num_dim = in_grad.shape.num_dims().unwrap_nonnegative();
  assert(info.num_dim == m.perm.size());

  LegionOrdered<legion_dim_t> legion_ordered_perm =
      legion_ordered_perm_from_ff_ordered(m.perm);

  for (int i = 0; i < info.num_dim; i++) {
    if (i == 0) {
      info.in_strides[i] = 1;
      info.out_strides[i] = 1;
    } else {
      int in_dim_size = out_grad.shape.at(legion_dim_t{nonnegative_int{i}})
                            .unwrap_nonnegative();
      int out_dim_size = in_grad.shape.at(legion_dim_t{nonnegative_int{i}})
                             .unwrap_nonnegative();
      info.in_strides[i] = info.in_strides[i - 1] * in_dim_size;
      info.out_strides[i] = info.out_strides[i - 1] * out_dim_size;
    }
    info.perm[legion_ordered_perm.at(legion_dim_t{nonnegative_int{i}})
                  .value.unwrap_nonnegative()] = i;
  }
  transpose_simple_kernel<<<
      GET_BLOCKS(in_grad.shape.get_volume().unwrap_nonnegative()),
      CUDA_NUM_THREADS,
      0,
      stream>>>(in_grad.shape.get_volume().unwrap_nonnegative(),
                out_grad.get_float_ptr(),
                in_grad.get_float_ptr(),
                info,
                1.0f /*beta*/);
}

} // namespace Transpose
} // namespace Kernels
} // namespace FlexFlow
