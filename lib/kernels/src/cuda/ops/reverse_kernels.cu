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
#include "kernels/reverse_kernels.h"
#include "kernels/reverse_kernels_params.h"

namespace FlexFlow::Kernels::Reverse {

__global__ void reverse_forward_kernel(float const *in_ptr,
                                       float *out_ptr,
                                       coord_t num_out_blks,
                                       coord_t reverse_dim_size,
                                       coord_t in_blk_size) {
  CUDA_KERNEL_LOOP(i, num_out_blks * reverse_dim_size * in_blk_size) {
    coord_t out_idx = i;
    coord_t blk_idx = i / (reverse_dim_size * in_blk_size);
    i = i - blk_idx * (reverse_dim_size * in_blk_size);
    coord_t reverse_dim_idx = i / in_blk_size;
    i = i - reverse_dim_idx * in_blk_size;
    coord_t in_idx = blk_idx * (reverse_dim_size * in_blk_size) +
                     (reverse_dim_size - 1 - reverse_dim_idx) * in_blk_size + i;
    out_ptr[out_idx] = in_ptr[in_idx];
  }
}

static void forward_kernel_internal(cudaStream_t stream,
                                    float const *in_ptr,
                                    float *out_ptr,
                                    coord_t num_out_blks,
                                    coord_t reverse_dim_size,
                                    coord_t in_blk_size,
                                    coord_t output_size) {

  reverse_forward_kernel<<<GET_BLOCKS(output_size),
                           CUDA_NUM_THREADS,
                           0,
                           stream>>>(
      in_ptr, out_ptr, num_out_blks, reverse_dim_size, in_blk_size);
}

void forward_kernel(ffStream_t stream,
                    GenericTensorAccessorR const &input_accessor,
                    GenericTensorAccessorW &output_accessor,
                    ReverseAttrs const &attrs) {

  auto reverse_kernels_params =
      compute_reverse_kernels_params(output_accessor.shape, attrs);

  forward_kernel_internal(
      stream,
      input_accessor.get_float_ptr(),
      output_accessor.get_float_ptr(),
      reverse_kernels_params.num_out_blks.unwrap_nonnegative(),
      reverse_kernels_params.reverse_dim_size.unwrap_nonnegative(),
      reverse_kernels_params.in_blk_size.unwrap_nonnegative(),
      reverse_kernels_params.out_size.unwrap_nonnegative());
}

void backward_kernel_internal(cudaStream_t stream,
                              float const *out_grad_ptr,
                              float *in_grad_ptr,
                              coord_t num_out_blks,
                              coord_t reverse_dim_size,
                              coord_t in_blk_size,
                              coord_t input_size) {

  reverse_forward_kernel<<<GET_BLOCKS(input_size),
                           CUDA_NUM_THREADS,
                           0,
                           stream>>>(
      out_grad_ptr, in_grad_ptr, num_out_blks, reverse_dim_size, in_blk_size);
}

void backward_kernel(ffStream_t stream,
                     GenericTensorAccessorR const &output_grad_accessor,
                     GenericTensorAccessorW &input_grad_accessor,
                     ReverseAttrs const &attrs) {
  auto reverse_kernels_params =
      compute_reverse_kernels_params(input_grad_accessor.shape, attrs);

  backward_kernel_internal(
      stream,
      output_grad_accessor.get_float_ptr(),
      input_grad_accessor.get_float_ptr(),
      reverse_kernels_params.num_out_blks.unwrap_nonnegative(),
      reverse_kernels_params.reverse_dim_size.unwrap_nonnegative(),
      reverse_kernels_params.in_blk_size.unwrap_nonnegative(),
      reverse_kernels_params.out_size.unwrap_nonnegative());
}

} // namespace FlexFlow::Kernels::Reverse
