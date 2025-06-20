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
#include "kernels/replicate_kernels_gpu.h"

namespace FlexFlow {
namespace Kernels {
namespace Replicate {

template <typename T>
__global__ void replicate_backward_kernel(T const *output_ptr,
                                          T *input_ptr,
                                          size_t num_elements,
                                          size_t num_replicas) {
  CUDA_KERNEL_LOOP(i, num_elements) {
    for (size_t j = 0; j < num_replicas; j++) {
      input_ptr[i] += output_ptr[i + j * num_elements];
    }
  }
}

template <DataType T>
struct ForwardKernel {
  void operator()(cudaStream_t stream,
                  GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &output) {
    checkCUDA(
        cudaMemcpyAsync((void *)output.get<T>(),
                        (void *)input.get<T>(),
                        input.shape.num_elements().int_from_positive_int() *
                            size_of_datatype(T).int_from_positive_int(),
                        cudaMemcpyDeviceToDevice,
                        stream));
  }
};

template <DataType T>
struct BackwardKernel {
  void operator()(cudaStream_t stream,
                  GenericTensorAccessorR const &output,
                  GenericTensorAccessorW const &input,
                  size_t num_replicas) {
    size_t total_elements =
        input.shape.num_elements().int_from_positive_int() * num_replicas;
    replicate_backward_kernel<real_type_t<T>>
        <<<GET_BLOCKS(total_elements), CUDA_NUM_THREADS, 0, stream>>>(
            output.get<T>(),
            input.get<T>(),
            input.shape.num_elements().int_from_positive_int(),
            num_replicas);
  }
};

void gpu_forward_kernel(cudaStream_t stream,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output) {
  DataTypeDispatch1<ForwardKernel>{}(input.data_type, stream, input, output);
}

void gpu_backward_kernel(cudaStream_t stream,
                     GenericTensorAccessorR const &output,
                     GenericTensorAccessorW const &input,
                     size_t num_replicas) {
  DataTypeDispatch1<BackwardKernel>{}(
      input.data_type, stream, output, input, num_replicas);
}

} // namespace Replicate
} // namespace Kernels
} // namespace FlexFlow
