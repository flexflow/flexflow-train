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

#include "kernels/flat_kernels.h"
#include "internal/device.h"
#include "kernels/accessor.h"
#include <hip/hip_runtime.h>

namespace FlexFlow {
namespace Kernels {
namespace Flat {

void forward_kernel(hipStream_t stream,
                    GenericTensorAccessorR input,
                    float *output_ptr) {

  checkCUDA(hipMemcpyAsync(output_ptr,
                           input.get_float_ptr(),
                           (input.shape.num_elements()) * sizeof(float),
                           hipMemcpyDeviceToDevice,
                           stream));
}

void backward_kernel(hipStream_t stream,
                     GenericTensorAccessorR input,
                     float *input_grad_ptr,
                     float const *output_grad_ptr) {

  float alpha = 1.0f;
  hipLaunchKernelGGL(HIP_KERNEL_NAME(apply_add_with_scale<float>),
                     GET_BLOCKS(input.shape.num_elements()),
                     CUDA_NUM_THREADS,
                     0,
                     stream,
                     input_grad_ptr,
                     output_grad_ptr,
                     input.shape.num_elements(),
                     alpha);
}

} // namespace Flat
} // namespace Kernels
} // namespace FlexFlow
