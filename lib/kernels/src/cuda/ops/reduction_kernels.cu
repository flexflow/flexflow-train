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
#include "kernels/datatype_dispatch.h"
#include "kernels/reduction_kernels_gpu.h"

namespace FlexFlow {
namespace Kernels {
namespace Reduction {

template <typename T>
__global__ void reduction_forward_kernel(T const *input_ptr,
                                         T *output_ptr,
                                         size_t num_elements,
                                         size_t num_replicas) {
  CUDA_KERNEL_LOOP(i, num_elements) {
    output_ptr[i] = input_ptr[i];
    for (size_t j = 1; j < num_replicas; j++) {
      output_ptr[i] += input_ptr[i + j * num_elements];
    }
  }
}

template <DataType T>
struct ForwardKernel {
  void operator()(cudaStream_t stream,
                  GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &output,
                  size_t num_replicas) {

    size_t total_elements =
        get_num_elements(input.shape.dims).int_from_positive_int() * num_replicas;
    reduction_forward_kernel<real_type_t<T>>
        <<<GET_BLOCKS(total_elements), CUDA_NUM_THREADS, 0, stream>>>(
            input.get<T>(),
            output.get<T>(),
            get_num_elements(input.shape.dims).int_from_positive_int(),
            num_replicas);
  }
};

void gpu_forward_kernel(cudaStream_t stream,
                        GenericTensorAccessorR const &input,
                        GenericTensorAccessorW const &output,
                        size_t num_replicas) {
  DataTypeDispatch1<ForwardKernel>{}(
      input.shape.data_type, stream, input, output, num_replicas);
}

void gpu_backward_kernel(cudaStream_t stream,
                         GenericTensorAccessorR const &output,
                         GenericTensorAccessorW const &input) {
  copy_accessor_data_to_l_from_r(input, output);
}

} // namespace Reduction
} // namespace Kernels
} // namespace FlexFlow
