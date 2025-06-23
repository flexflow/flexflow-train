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
#include "kernels/cast_kernels_gpu.h"
#include "kernels/datatype_dispatch.h"

namespace FlexFlow {
namespace Kernels {
namespace Cast {

template <typename IDT, typename ODT>
__global__ void cast_forward(IDT const *input, ODT *output, size_t volume) {
  CUDA_KERNEL_LOOP(i, volume) {
    output[i] = (ODT)input[i];
  }
}

template <typename IDT, typename ODT>
__global__ void
    cast_backward(IDT const *input, ODT *output, size_t volume, ODT beta) {
  CUDA_KERNEL_LOOP(i, volume) {
    output[i] = (ODT)input[i] + beta * output[i];
  }
}

template <DataType IDT, DataType ODT>
struct ForwardKernel {
  void operator()(ffStream_t stream,
                  GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &output) {
    size_t volume = input.shape.num_elements().int_from_positive_int();
    cast_forward<<<GET_BLOCKS(volume), CUDA_NUM_THREADS, 0, stream>>>(
        input.get<IDT>(), output.get<ODT>(), volume);
  }
};

template <DataType IDT, DataType ODT>
struct BackwardKernel {
  void operator()(ffStream_t stream,
                  GenericTensorAccessorR const &output,
                  GenericTensorAccessorW const &input) {
    size_t volume = output.shape.num_elements().int_from_positive_int();
    cast_backward<<<GET_BLOCKS(volume), CUDA_NUM_THREADS, 0, stream>>>(
        output.get<IDT>(), input.get<ODT>(), volume, cast_to<ODT>(1.0f));
  }
};

void gpu_forward_kernel(ffStream_t stream,
                        GenericTensorAccessorR const &input,
                        GenericTensorAccessorW const &output) {
  DataTypeDispatch2<ForwardKernel>{}(
      input.data_type, output.data_type, stream, input, output);
}

void gpu_backward_kernel(ffStream_t stream,
                         GenericTensorAccessorR const &output,
                         GenericTensorAccessorW const &input) {
  DataTypeDispatch2<BackwardKernel>{}(
      output.data_type, input.data_type, stream, output, input);
}

} // namespace Cast
} // namespace Kernels
} // namespace FlexFlow
