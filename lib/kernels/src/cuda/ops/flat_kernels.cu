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
#include "kernels/flat_kernels_gpu.h"
#include "op-attrs/tensor_shape.h"

namespace FlexFlow {
namespace Kernels {
namespace Flat {

void gpu_forward_kernel(cudaStream_t stream,
                        GenericTensorAccessorR const &input,
                        float *output_ptr) {

  checkCUDA(cudaMemcpyAsync(output_ptr,
                            input.get_float_ptr(),
                            get_size_in_bytes(input.shape).unwrap_num_bytes().unwrap_nonnegative(),
                            cudaMemcpyDeviceToDevice,
                            stream));
}

void gpu_backward_kernel(cudaStream_t stream,
                         GenericTensorAccessorR const &input,
                         float const *output_grad_ptr,
                         float *input_grad_ptr) {

  float alpha = 1.0f;
  apply_add_with_scale<float>
      <<<GET_BLOCKS(get_num_elements(input.shape.dims).int_from_positive_int()),
         CUDA_NUM_THREADS,
         0,
         stream>>>(input_grad_ptr,
                   output_grad_ptr,
                   get_num_elements(input.shape.dims).int_from_positive_int(),
                   alpha);
}

} // namespace Flat
} // namespace Kernels
} // namespace FlexFlow
