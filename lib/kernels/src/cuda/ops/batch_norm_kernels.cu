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
#include "kernels/allocation.h"
#include "kernels/batch_norm_kernels_gpu.h"
#include "kernels/ff_handle.h"
#include "utils/integer_conversions.h"
#include "utils/optional.h"
#include <algorithm>
#include <vector>

namespace FlexFlow {
namespace Kernels {
namespace BatchNorm {

void gpu_forward_kernel(cudaStream_t stream,
                        BatchNormPerDeviceState const &m,
                        float const *input_ptr,
                        float *output_ptr,
                        float const *scale_ptr,
                        float const *bias_ptr) {
  checkCUDNN(cudnnSetStream(m.handle.dnn, stream));

  float alpha = 1.0f, beta = 0.0f;
  checkCUDNN(cudnnBatchNormalizationForwardTraining(m.handle.dnn,
                                                    m.mode,
                                                    &alpha,
                                                    &beta,
                                                    m.inputTensor,
                                                    input_ptr,
                                                    m.outputTensor,
                                                    output_ptr,
                                                    m.biasTensor,
                                                    scale_ptr,
                                                    bias_ptr,
                                                    1.0,
                                                    m.runningMean,
                                                    m.runningVar,
                                                    m.eps,
                                                    m.saveMean,
                                                    m.saveVar));
  if (m.activation.has_value()) {
    checkCUDNN(cudnnActivationForward(m.handle.dnn,
                                      m.actiDesc,
                                      &alpha,
                                      m.outputTensor,
                                      output_ptr,
                                      &beta,
                                      m.outputTensor,
                                      output_ptr));
  }
}

void gpu_backward_kernel(cudaStream_t stream,
                         BatchNormPerDeviceState const &m,
                         float const *output_ptr,
                         float *output_grad_ptr,
                         float const *input_ptr,
                         float *input_grad_ptr,
                         float const *scale_ptr,
                         float *scale_grad_ptr,
                         float *bias_grad_ptr,
                         size_t numElements) {
  checkCUDNN(cudnnSetStream(m.handle.dnn, stream));

  float alpha = 1.0f, beta = 0.0f;
  if (m.activation.has_value()) {
    switch (m.activation.value()) {
      case Activation::RELU:
        reluBackward<<<GET_BLOCKS(numElements), CUDA_NUM_THREADS, 0, stream>>>(
            output_grad_ptr, output_ptr, numElements);
        break;
      default:
        PANIC("unsupported activation function for BatchNorm",
              m.activation.value());
    }
  }

  checkCUDNN(cudnnBatchNormalizationBackward(m.handle.dnn,
                                             m.mode,
                                             &alpha,
                                             &beta,
                                             &alpha,
                                             &beta,
                                             m.inputTensor,
                                             input_ptr,
                                             m.outputTensor,
                                             output_grad_ptr,
                                             m.inputTensor,
                                             input_grad_ptr,
                                             m.biasTensor,
                                             scale_ptr,
                                             scale_grad_ptr,
                                             bias_grad_ptr,
                                             m.eps,
                                             m.saveMean,
                                             m.saveVar));
}

BatchNormPerDeviceState
    gpu_init_kernel(PerDeviceFFHandle const &handle,
                    Allocator &allocator,
                    float *runningMean,
                    int output_n,
                    int output_c,
                    int output_h,
                    int output_w,
                    std::optional<Activation> const &activation,
                    float eps) {
  ASSERT(eps >= static_cast<float>(CUDNN_BN_MIN_EPSILON),
         "cuDNN requires BatchNorm eps to be at least CUDNN_BN_MIN_EPSILON",
         eps);
  ffTensorDescriptor_t inputTensor;
  ffTensorDescriptor_t outputTensor;
  ffTensorDescriptor_t biasTensor;
  ffActivationDescriptor_t actiDesc{};
  ffBatchNormMode_t mode;
  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&biasTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  mode = CUDNN_BATCHNORM_SPATIAL;
#if CUDNN_VERSION >= 7000
  mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
#endif
  checkCUDNN(cudnnSetTensor4dDescriptor(inputTensor,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        output_n,
                                        output_c,
                                        output_h,
                                        output_w));
  checkCUDNN(cudnnSetTensor4dDescriptor(outputTensor,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        output_n,
                                        output_c,
                                        output_h,
                                        output_w));
  checkCUDNN(cudnnSetTensor4dDescriptor(
      biasTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, output_c, 1, 1));
  // allocate memory for runningMean, runningVar, saveMean, saveVar
  size_t totalSize = sizeof(float) * output_c * 4;
  runningMean = (float *)allocator.allocate(totalSize);
  float *runningVar = (float *)runningMean + output_c;
  float *saveMean = (float *)runningVar + output_c;
  float *saveVar = (float *)saveMean + output_c;
  // Initialize on the host so standalone kernels do not need a Legion stream.
  std::vector<float> initial_running_stats(output_c * 2, 0.0f);
  std::fill(initial_running_stats.begin() + output_c,
            initial_running_stats.end(),
            1.0f);
  checkCUDA(cudaMemcpy(runningMean,
                       initial_running_stats.data(),
                       sizeof(float) * output_c * 2,
                       cudaMemcpyHostToDevice));

  if (activation.has_value()) {
    switch (activation.value()) {
      case Activation::RELU:
        checkCUDNN(cudnnCreateActivationDescriptor(&actiDesc));
        checkCUDNN(cudnnSetActivationDescriptor(
            actiDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));
        break;
      default:
        PANIC("unsupported activation function for BatchNorm",
              activation.value());
    }
  }

  BatchNormPerDeviceState per_device_state = BatchNormPerDeviceState{
      handle,
      inputTensor,
      outputTensor,
      biasTensor,
      actiDesc,
      mode,
      runningMean,
      runningVar,
      saveMean,
      saveVar,
      output_n,
      output_c,
      output_h,
      output_w,
      activation,
      // A float epsilon of 1e-5 rounds just below cuDNN's double minimum.
      std::max(static_cast<double>(eps), CUDNN_BN_MIN_EPSILON),
  };

  return per_device_state;
}

void gpu_cleanup_kernel(Allocator &allocator,
                        BatchNormPerDeviceState &per_device_state) {
  allocator.deallocate(per_device_state.runningMean);
  checkCUDNN(cudnnDestroyTensorDescriptor(per_device_state.inputTensor));
  checkCUDNN(cudnnDestroyTensorDescriptor(per_device_state.biasTensor));
  checkCUDNN(cudnnDestroyTensorDescriptor(per_device_state.outputTensor));

  if (per_device_state.activation.has_value()) {
    Activation activation = assert_unwrap(per_device_state.activation);
    switch (activation) {
      case Activation::RELU:
        checkCUDNN(cudnnDestroyActivationDescriptor(per_device_state.actiDesc));
        break;
      default:
        PANIC("unsupported activation function for BatchNorm", activation);
    }
  }
}

} // namespace BatchNorm
} // namespace Kernels
} // namespace FlexFlow
