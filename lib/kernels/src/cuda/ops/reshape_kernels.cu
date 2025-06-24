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
#include "kernels/reshape_kernels_gpu.h"

namespace FlexFlow {

namespace Kernels {
namespace Reshape {

template <DataType InputDT, DataType OutputDT>
struct ForwardKernel {
  void operator()(cudaStream_t stream,
                  GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &output) {
    checkCUDA(
        cudaMemcpyAsync(output.get<OutputDT>(),
                        input.get<InputDT>(),
                        input.shape.num_elements().int_from_positive_int() *
                            size_of_datatype(OutputDT).int_from_positive_int(),
                        cudaMemcpyDeviceToDevice,
                        stream));
  }
};

template <typename DT, typename DTGrad>
__global__ void apply_add_with_scale2(DT *data_ptr,
                                      DTGrad const *grad_ptr,
                                      size_t size,
                                      DT scale) {
  CUDA_KERNEL_LOOP(i, size) {
    data_ptr[i] += grad_ptr[i] * scale;
  }
}

template <DataType InputDT, DataType OutputDT>
struct BackwardKernel {
  void operator()(cudaStream_t stream,
                  GenericTensorAccessorR const &output,
                  GenericTensorAccessorW const &input) {
    float alpha = 1.0f;
    apply_add_with_scale2<real_type_t<InputDT>, real_type_t<OutputDT>>
        <<<GET_BLOCKS(input.shape.num_elements().int_from_positive_int()),
           CUDA_NUM_THREADS,
           0,
           stream>>>(input.get<InputDT>(),
                     output.get<OutputDT>(),
                     input.shape.num_elements().int_from_positive_int(),
                     static_cast<real_type_t<InputDT>>(alpha));
  }
};

void gpu_forward_kernel(cudaStream_t stream,
                        GenericTensorAccessorR const &input,
                        GenericTensorAccessorW const &output) {
  DataTypeDispatch2<ForwardKernel>{}(
      input.data_type, output.data_type, stream, input, output);
}

void gpu_backward_kernel(cudaStream_t stream,
                         GenericTensorAccessorR const &output,
                         GenericTensorAccessorW const &input) {
  DataTypeDispatch2<BackwardKernel>{}(
      input.data_type, output.data_type, stream, output, input);
}

} // namespace Reshape
} // namespace Kernels
} // namespace FlexFlow
