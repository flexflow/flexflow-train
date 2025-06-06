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

#include "kernels/transpose_kernels.h"
#include "internal/device.h"
#include "kernels/accessor.h"
#include "utils/exception.h"
#include <hip/hip_runtime.h>

namespace FlexFlow {

struct TransposeStrides {
  int num_dim;
  int in_strides[MAX_TENSOR_DIM], out_strides[MAX_TENSOR_DIM],
      perm[MAX_TENSOR_DIM];
};

namespace Kernels {
namespace Transpose {

TransposePerDeviceState init_kernel(int num_dim,
                                    std::vector<ff_dim_t> const &perm) {
  int const length = perm.size();

  std::vector<int> perm_vector;
  assert(length <= MAX_TENSOR_DIM);
  for (int i = 0; i < length; ++i) {
    perm_vector.push_back(perm[i].value());
  }

  return {num_dim, perm_vector};
}

__global__ void transpose_simple_kernel(std::size_t volume,
                                        float const *in_ptr,
                                        float *out_ptr,
                                        const TransposeStrides info,
                                        float const beta) {
  CUDA_KERNEL_LOOP(o_idx, volume) {
    coord_t i_idx = 0;
    coord_t t = o_idx;
    for (int i = info.num_dim - 1; i >= 0; i--) {
      coord_t ratio = t / info.out_strides[i];
      t -= ratio * info.out_strides[i];
      i_idx += ratio * info.in_strides[info.perm[i]];
    }
    out_ptr[o_idx] += out_ptr[o_idx] * beta + in_ptr[i_idx];
  }
}

void forward_kernel(hipStream_t stream,
                    TransposePerDeviceState const &m,
                    GenericTensorAccessorW const &in_grad,
                    GenericTensorAccessorR const &out_grad) {

  TransposeStrides info;
  info.num_dim = in_grad.shape.num_dims();
  assert(info.num_dim == m.num_dim);
  for (int i = 0; i < info.num_dim; i++) {
    if (i == 0) {
      info.in_strides[i] = 1;
      info.out_strides[i] = 1;
    } else {
      int in_dim_size = input.shape[legion_dim_t(i)] + 1;
      int out_dim_size = output.shape[legion_dim_t(i)] + 1;
      info.in_strides[i] = info.in_strides[i - 1] * in_dim_size;
      info.out_strides[i] = info.out_strides[i - 1] * out_dim_size;
    }
    info.perm[i] = m.perm[i];
  }

  hipLaunchKernelGGL(transpose_simple_kernel,
                     GET_BLOCKS(output.shape.get_volume()),
                     CUDA_NUM_THREADS,
                     0,
                     stream,
                     output.shape.get_volume(),
                     input.get_float_ptr(),
                     output.get_float_ptr(),
                     info,
                     0.0f /*beta*/);
}

void backward_kernel(hipStream_t stream,
                     TransposePerDeviceState const &m,
                     float *input_grad_ptr,
                     float const *output_grad_ptr,
                     Domain in_grad_domain,
                     Domain out_grad_domain) {

  TransposeStrides info;
  info.num_dim = in_grad.shape.num_dims();
  assert(info.num_dim == m.num_dim);
  for (int i = 0; i < info.num_dim; i++) {
    if (i == 0) {
      info.in_strides[i] = 1;
      info.out_strides[i] = 1;
    } else {
      int in_dim_size = out_grad.shape[legion_dim_t(i)] + 1;
      int out_dim_size = in_grad.shape[legion_dim_t(i)] + 1;
      info.in_strides[i] = info.in_strides[i - 1] * in_dim_size;
      info.out_strides[i] = info.out_strides[i - 1] * out_dim_size;
    }
    info.perm[m.perm[i]] = i;
  }
  hipLaunchKernelGGL(transpose_simple_kernel,
                     GET_BLOCKS(in_grad.shape.get_volume()),
                     CUDA_NUM_THREADS,
                     0,
                     stream,
                     in_grad.shape.get_volume(),
                     out_grad.get_float_ptr(),
                     in_grad.get_float_ptr(),
                     info,
                     1.0f /*beta*/);
}

} // namespace Transpose
} // namespace Kernels
} // namespace FlexFlow
