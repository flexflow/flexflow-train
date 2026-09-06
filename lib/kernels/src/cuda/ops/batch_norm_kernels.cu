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
#include "kernels/batch_norm_kernels_gpu.h"
#include "op-attrs/ops/batch_norm.h"
#include "op-attrs/tensor_dims.h"
#include "utils/containers/require_same.h"
#include <vector>

namespace FlexFlow {

static positive_int get_num_channels(TensorShape const &shape) {
  return dim_at_idx(shape.dims, ff_dim_t{1_n});
}

namespace {

/**
 * @brief An NCHW shape as the fused kernels index it: one block per channel,
 * walking the samples of that channel in turn.
 */
struct BatchNormExtents {
  int num_samples;
  int num_channels;
  int spatial_size;
};

BatchNormExtents get_extents(TensorShape const &shape) {
  return BatchNormExtents{
      /*num_samples=*/dim_at_idx(shape.dims, ff_dim_t{0_n})
          .int_from_positive_int(),
      /*num_channels=*/
      dim_at_idx(shape.dims, ff_dim_t{1_n}).int_from_positive_int(),
      /*spatial_size=*/
      (dim_at_idx(shape.dims, ff_dim_t{2_n}) *
       dim_at_idx(shape.dims, ff_dim_t{3_n}))
          .int_from_positive_int(),
  };
}

} // namespace

namespace {

/**
 * @brief The number of threads in each of the fused kernels' blocks.
 *
 * @details Each block reduces one channel, so this is also how much parallelism
 * a channel gets. Measured across every batch-norm site in YOLOv10x, 512 beat
 * both 256 (+32% forward, +21% backward) and 1024 (+8%, +5%): below it the GPU
 * runs out of threads to hide memory latency with, above it the block-wide
 * reductions start to cost more than the occupancy buys.
 */
constexpr int BATCH_NORM_FUSED_THREADS = 512;

__device__ __forceinline__ float apply_activation(Activation activation,
                                                  float y) {
  switch (activation) {
    case Activation::RELU:
      return y > 0.0f ? y : 0.0f;
    case Activation::SIGMOID:
      return 1.0f / (1.0f + expf(-y));
    case Activation::TANH:
      return tanhf(y);
    case Activation::SILU:
      return y / (1.0f + expf(-y));
    default:
      return y;
  }
}

/**
 * @brief The gradient with respect to an activation's input, given \p dz, the
 * gradient with respect to its output, and \p y, its input.
 */
__device__ __forceinline__ float
    activation_backward(Activation activation, float dz, float y) {
  switch (activation) {
    case Activation::RELU:
      return y > 0.0f ? dz : 0.0f;
    case Activation::SIGMOID: {
      float s = 1.0f / (1.0f + expf(-y));
      return dz * s * (1.0f - s);
    }
    case Activation::TANH: {
      float t = tanhf(y);
      return dz * (1.0f - t * t);
    }
    case Activation::SILU: {
      // Written in terms of the sigmoid rather than the algebraically equal
      // e^y (y + e^y + 1) / (e^y + 1)^2, which evaluates to NaN once y is much
      // above 88 and e^y overflows. See element_unary_kernels.cu, which this
      // has to agree with.
      float s = 1.0f / (1.0f + expf(-y));
      return dz * s * (1.0f + y * (1.0f - s));
    }
    default:
      return dz;
  }
}

__device__ __forceinline__ void block_reduce_sum2(float &a, float &b) {
  __shared__ float shared_a[BATCH_NORM_FUSED_THREADS / 32];
  __shared__ float shared_b[BATCH_NORM_FUSED_THREADS / 32];
  int lane = threadIdx.x % 32;
  int warp = threadIdx.x / 32;

  for (int offset = 16; offset > 0; offset >>= 1) {
    a += __shfl_down_sync(0xffffffff, a, offset);
    b += __shfl_down_sync(0xffffffff, b, offset);
  }
  if (lane == 0) {
    shared_a[warp] = a;
    shared_b[warp] = b;
  }
  __syncthreads();

  if (warp == 0) {
    bool in_range = threadIdx.x < BATCH_NORM_FUSED_THREADS / 32;
    a = in_range ? shared_a[threadIdx.x] : 0.0f;
    b = in_range ? shared_b[threadIdx.x] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      a += __shfl_down_sync(0xffffffff, a, offset);
      b += __shfl_down_sync(0xffffffff, b, offset);
    }
    if (threadIdx.x == 0) {
      shared_a[0] = a;
      shared_b[0] = b;
    }
  }
  __syncthreads();
  a = shared_a[0];
  b = shared_b[0];
}

/**
 * @brief Fold one set of Welford statistics into another.
 *
 * @details Welford's method is used in place of the textbook
 * <tt>E[x^2] - E[x]^2</tt> because that form subtracts two numbers that are
 * nearly equal whenever the mean is large next to the standard deviation,
 * which loses most of the significant digits and can even come out negative,
 * making the inverse standard deviation NaN.
 */
__device__ __forceinline__ void welford_join(
    float &count, float &mean, float &m2, float n, float m, float s) {
  float total = count + n;
  if (total == 0.0f) {
    return;
  }
  float delta = m - mean;
  mean += delta * (n / total);
  m2 += s + delta * delta * (count * n / total);
  count = total;
}

__device__ __forceinline__ void
    block_reduce_welford(float &count, float &mean, float &m2) {
  __shared__ float shared_count[BATCH_NORM_FUSED_THREADS / 32];
  __shared__ float shared_mean[BATCH_NORM_FUSED_THREADS / 32];
  __shared__ float shared_m2[BATCH_NORM_FUSED_THREADS / 32];
  int lane = threadIdx.x % 32;
  int warp = threadIdx.x / 32;

  for (int offset = 16; offset > 0; offset >>= 1) {
    welford_join(count,
                 mean,
                 m2,
                 __shfl_down_sync(0xffffffff, count, offset),
                 __shfl_down_sync(0xffffffff, mean, offset),
                 __shfl_down_sync(0xffffffff, m2, offset));
  }
  if (lane == 0) {
    shared_count[warp] = count;
    shared_mean[warp] = mean;
    shared_m2[warp] = m2;
  }
  __syncthreads();

  if (warp == 0) {
    bool in_range = threadIdx.x < BATCH_NORM_FUSED_THREADS / 32;
    count = in_range ? shared_count[threadIdx.x] : 0.0f;
    mean = in_range ? shared_mean[threadIdx.x] : 0.0f;
    m2 = in_range ? shared_m2[threadIdx.x] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      welford_join(count,
                   mean,
                   m2,
                   __shfl_down_sync(0xffffffff, count, offset),
                   __shfl_down_sync(0xffffffff, mean, offset),
                   __shfl_down_sync(0xffffffff, m2, offset));
    }
    if (threadIdx.x == 0) {
      shared_count[0] = count;
      shared_mean[0] = mean;
      shared_m2[0] = m2;
    }
  }
  __syncthreads();
  count = shared_count[0];
  mean = shared_mean[0];
  m2 = shared_m2[0];
}

/**
 * @brief Normalize and activate in one pass over the input, one block per
 * channel.
 *
 * @details Saves the two passes over the output that a separate activation
 * operator would cost: nothing ever writes or reads the normalized value, which
 * exists only in registers. \ref batch_norm_fused_backward_kernel reconstructs
 * it from \p input rather than being handed it, so it is not needed later
 * either.
 */
template <Activation ACTIVATION>
__global__ void
    batch_norm_fused_forward_kernel(int num_samples,
                                    int num_channels,
                                    int spatial_size,
                                    float eps,
                                    float exponential_average_factor,
                                    float const *input,
                                    float const *gamma,
                                    float const *beta,
                                    float *output,
                                    float *save_mean,
                                    float *save_invstd,
                                    float *running_mean,
                                    float *running_var) {
  int channel = blockIdx.x;

  float count = 0.0f, mean = 0.0f, m2 = 0.0f;
  for (int sample = 0; sample < num_samples; sample++) {
    size_t base = ((size_t)sample * num_channels + channel) * spatial_size;
    for (int i = threadIdx.x; i < spatial_size; i += BATCH_NORM_FUSED_THREADS) {
      welford_join(count, mean, m2, 1.0f, input[base + i], 0.0f);
    }
  }
  block_reduce_welford(count, mean, m2);

  float variance = m2 / count;
  float invstd = rsqrtf(variance + eps);

  if (threadIdx.x == 0) {
    save_mean[channel] = mean;
    save_invstd[channel] = invstd;

    // Matching cuDNN, which keeps the running statistics in the units of the
    // data and uses the unbiased variance for them even though the biased one
    // is what normalizes this batch.
    float unbiased_variance = count > 1.0f ? m2 / (count - 1.0f) : variance;
    running_mean[channel] =
        (1.0f - exponential_average_factor) * running_mean[channel] +
        exponential_average_factor * mean;
    running_var[channel] =
        (1.0f - exponential_average_factor) * running_var[channel] +
        exponential_average_factor * unbiased_variance;
  }

  float scale = gamma[channel];
  float shift = beta[channel];
  for (int sample = 0; sample < num_samples; sample++) {
    size_t base = ((size_t)sample * num_channels + channel) * spatial_size;
    for (int i = threadIdx.x; i < spatial_size; i += BATCH_NORM_FUSED_THREADS) {
      float normalized = (input[base + i] - mean) * invstd;
      output[base + i] =
          apply_activation(ACTIVATION, scale * normalized + shift);
    }
  }
}

/**
 * @brief Differentiate the activation and the normalization together, one block
 * per channel.
 *
 * @details The value between the two, which a separate activation operator
 * would have read from memory, is instead reconstructed from \p input and the
 * statistics the forward pass saved. That trades one multiply-add for a whole
 * pass over the tensor, and is what lets the forward pass get away with never
 * writing it down.
 *
 * Like the cuDNN path, accumulates into the gradients rather than overwriting
 * them.
 */
template <Activation ACTIVATION>
__global__ void batch_norm_fused_backward_kernel(int num_samples,
                                                 int num_channels,
                                                 int spatial_size,
                                                 float const *output_grad,
                                                 float const *input,
                                                 float const *gamma,
                                                 float const *beta,
                                                 float const *save_mean,
                                                 float const *save_invstd,
                                                 float *input_grad,
                                                 float *gamma_grad,
                                                 float *beta_grad) {
  int channel = blockIdx.x;
  float mean = save_mean[channel];
  float invstd = save_invstd[channel];
  float scale = gamma[channel];
  float shift = beta[channel];

  float sum_grad = 0.0f, sum_grad_normalized = 0.0f;
  for (int sample = 0; sample < num_samples; sample++) {
    size_t base = ((size_t)sample * num_channels + channel) * spatial_size;
    for (int i = threadIdx.x; i < spatial_size; i += BATCH_NORM_FUSED_THREADS) {
      float normalized = (input[base + i] - mean) * invstd;
      float grad = activation_backward(
          ACTIVATION, output_grad[base + i], scale * normalized + shift);
      sum_grad += grad;
      sum_grad_normalized += grad * normalized;
    }
  }
  block_reduce_sum2(sum_grad, sum_grad_normalized);

  if (threadIdx.x == 0) {
    gamma_grad[channel] += sum_grad_normalized;
    beta_grad[channel] += sum_grad;
  }

  float count = (float)num_samples * spatial_size;
  for (int sample = 0; sample < num_samples; sample++) {
    size_t base = ((size_t)sample * num_channels + channel) * spatial_size;
    for (int i = threadIdx.x; i < spatial_size; i += BATCH_NORM_FUSED_THREADS) {
      float normalized = (input[base + i] - mean) * invstd;
      float grad = activation_backward(
          ACTIVATION, output_grad[base + i], scale * normalized + shift);
      input_grad[base + i] +=
          scale * invstd *
          (grad - (sum_grad + normalized * sum_grad_normalized) / count);
    }
  }
}

/**
 * @brief Call \p f with the activation as a template argument.
 *
 * @details The activation is fixed for the life of an operator, so resolving it
 * once here keeps the branch out of the innermost loop.
 */
template <typename F>
void dispatch_on_activation(Activation activation, F const &f) {
  switch (activation) {
    case Activation::RELU:
      f(std::integral_constant<Activation, Activation::RELU>{});
      return;
    case Activation::SIGMOID:
      f(std::integral_constant<Activation, Activation::SIGMOID>{});
      return;
    case Activation::TANH:
      f(std::integral_constant<Activation, Activation::TANH>{});
      return;
    case Activation::SILU:
      f(std::integral_constant<Activation, Activation::SILU>{});
      return;
    default:
      PANIC("BatchNorm cannot compute this activation itself", activation);
  }
}

} // namespace

BatchNormPerDeviceState
    batch_norm_gpu_init_kernel(Allocator &allocator,
                               BatchNormAttrs const &attrs,
                               TensorShape const &input_shape,
                               TensorShape const &output_shape) {
  if (attrs.activation.has_value()) {
    ASSERT(batch_norm_supports_fused_activation(attrs.activation.value()),
           "BatchNorm cannot compute this activation itself",
           attrs.activation.value());
    ASSERT(batch_norm_mode_supports_fused_activation(attrs.mode),
           "BatchNorm cannot compute a fused activation in this mode",
           attrs.mode);
  }

  ASSERT(attrs.affine,
         "BatchNorm currently only supports attrs.affine = true. "
         "If you need this feature, please create an issue.");

  TensorShape shape = require_same(input_shape, output_shape);

  ASSERT(get_num_dims(shape.dims) == num_tensor_dims_t{4_n},
         "BatchNorm currently only supports 4-dimensional (i.e., NCHW) "
         "tensors. If you need this feature, please create an issue.",
         shape);

  ASSERT(attrs.eps >= CUDNN_BN_MIN_EPSILON,
         "cuDNN requires BatchNorm eps to be at least CUDNN_BN_MIN_EPSILON",
         attrs.eps,
         CUDNN_BN_MIN_EPSILON);

  int num_channels = get_num_channels(shape).int_from_positive_int();

  ffTensorDescriptor_t inputTensor;
  ffTensorDescriptor_t outputTensor;
  ffTensorDescriptor_t biasTensor;

  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&biasTensor));

  ffBatchNormMode_t mode;
  switch (attrs.mode) {
    case BatchNormMode::PER_ACTIVATION:
      mode = CUDNN_BATCHNORM_PER_ACTIVATION;
      break;
    case BatchNormMode::SPATIAL:
      mode = CUDNN_BATCHNORM_SPATIAL;
      break;
    case BatchNormMode::SPATIAL_PERSISTENT:
      mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
      break;
    default:
      PANIC("Unknown BatchNormMode", attrs.mode);
  }

  checkCUDNN(cudnnSetTensorDescriptorFromTensorShape(inputTensor, input_shape));
  checkCUDNN(
      cudnnSetTensorDescriptorFromTensorShape(outputTensor, output_shape));
  checkCUDNN(cudnnSetTensor4dDescriptor(biasTensor,
                                        CUDNN_TENSOR_NCHW,
                                        ff_to_cudnn_datatype(shape.data_type),
                                        /*n=*/1,
                                        /*c=*/num_channels,
                                        /*h=*/1,
                                        /*w=*/1));

  // Allocate memory for runningMean, runningVar, saveMean and saveVar as a
  // single contiguous block (deallocated by batch_norm_gpu_cleanup_kernel).
  float *runningMean = static_cast<float *>(
      allocator.allocate(sizeof(float) * num_channels * 4));
  float *runningVar = runningMean + num_channels;
  float *saveMean = runningVar + num_channels;
  float *saveVar = saveMean + num_channels;

  // Match the PyTorch initialization of running_mean = 0 and running_var = 1.
  std::vector<float> initial_running_stats(num_channels * 2);
  std::fill(initial_running_stats.begin(),
            initial_running_stats.begin() + num_channels,
            0.0f);
  std::fill(initial_running_stats.begin() + num_channels,
            initial_running_stats.end(),
            1.0f);
  checkCUDA(cudaMemcpy(runningMean,
                       initial_running_stats.data(),
                       sizeof(float) * num_channels * 2,
                       cudaMemcpyHostToDevice));

  return BatchNormPerDeviceState{
      /*inputTensor=*/inputTensor,
      /*outputTensor=*/outputTensor,
      /*biasTensor=*/biasTensor,
      /*mode=*/mode,
      /*runningMean=*/runningMean,
      /*runningVar=*/runningVar,
      /*saveMean=*/saveMean,
      /*saveVar=*/saveVar,
  };
}

void batch_norm_gpu_forward_kernel(
    cudaStream_t stream,
    PerDeviceFFHandle const &handle,
    BatchNormPerDeviceState const &per_device_state,
    BatchNormAttrs const &attrs,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorR const &gamma,
    GenericTensorAccessorR const &beta,
    GenericTensorAccessorW const &output) {
  checkCUDNN(cudnnSetStream(handle.dnn, stream));

  // NOTE: attrs.momentum = std::nullopt means "use a cumulative moving
  // average", which cuDNN cannot express, so we fall back to fully replacing
  // the running statistics on each call. The running statistics are currently
  // never read back, so this only matters once inference mode is supported.
  double exponential_average_factor = attrs.momentum.value_or(1.0);

  if (attrs.activation.has_value()) {
    BatchNormExtents extents = get_extents(input.shape);

    dispatch_on_activation(attrs.activation.value(), [&](auto activation) {
      batch_norm_fused_forward_kernel<decltype(activation)::value>
          <<<extents.num_channels, BATCH_NORM_FUSED_THREADS, 0, stream>>>(
              extents.num_samples,
              extents.num_channels,
              extents.spatial_size,
              attrs.eps,
              static_cast<float>(exponential_average_factor),
              input.get_float_ptr(),
              gamma.get_float_ptr(),
              beta.get_float_ptr(),
              output.get_float_ptr(),
              per_device_state.saveMean,
              per_device_state.saveVar,
              per_device_state.runningMean,
              per_device_state.runningVar);
    });
    return;
  }

  float alpha = 1.0f, beta_coeff = 0.0f;
  checkCUDNN(
      cudnnBatchNormalizationForwardTraining(handle.dnn,
                                             per_device_state.mode,
                                             &alpha,
                                             &beta_coeff,
                                             per_device_state.inputTensor,
                                             input.ptr,
                                             per_device_state.outputTensor,
                                             output.ptr,
                                             per_device_state.biasTensor,
                                             gamma.ptr,
                                             beta.ptr,
                                             exponential_average_factor,
                                             per_device_state.runningMean,
                                             per_device_state.runningVar,
                                             attrs.eps,
                                             per_device_state.saveMean,
                                             per_device_state.saveVar));
}

void batch_norm_gpu_backward_kernel(
    cudaStream_t stream,
    PerDeviceFFHandle const &handle,
    BatchNormPerDeviceState const &per_device_state,
    BatchNormAttrs const &attrs,
    GenericTensorAccessorR const &output,
    GenericTensorAccessorR const &output_grad,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &input_grad,
    GenericTensorAccessorR const &gamma,
    GenericTensorAccessorR const &beta,
    GenericTensorAccessorW const &gamma_grad,
    GenericTensorAccessorW const &beta_grad) {
  checkCUDNN(cudnnSetStream(handle.dnn, stream));

  if (attrs.activation.has_value()) {
    BatchNormExtents extents = get_extents(input.shape);

    dispatch_on_activation(attrs.activation.value(), [&](auto activation) {
      batch_norm_fused_backward_kernel<decltype(activation)::value>
          <<<extents.num_channels, BATCH_NORM_FUSED_THREADS, 0, stream>>>(
              extents.num_samples,
              extents.num_channels,
              extents.spatial_size,
              output_grad.get_float_ptr(),
              input.get_float_ptr(),
              gamma.get_float_ptr(),
              beta.get_float_ptr(),
              per_device_state.saveMean,
              per_device_state.saveVar,
              input_grad.get_float_ptr(),
              gamma_grad.get_float_ptr(),
              beta_grad.get_float_ptr());
    });
    return;
  }

  // NOTE: the beta coefficients are 1.0 so that the gradients are accumulated
  // into rather than overwritten
  float alpha_data = 1.0f, beta_data = 1.0f;
  float alpha_param = 1.0f, beta_param = 1.0f;
  checkCUDNN(cudnnBatchNormalizationBackward(handle.dnn,
                                             per_device_state.mode,
                                             &alpha_data,
                                             &beta_data,
                                             &alpha_param,
                                             &beta_param,
                                             per_device_state.inputTensor,
                                             input.ptr,
                                             per_device_state.outputTensor,
                                             output_grad.ptr,
                                             per_device_state.inputTensor,
                                             input_grad.ptr,
                                             per_device_state.biasTensor,
                                             gamma.ptr,
                                             gamma_grad.ptr,
                                             beta_grad.ptr,
                                             attrs.eps,
                                             per_device_state.saveMean,
                                             per_device_state.saveVar));
}

void batch_norm_gpu_cleanup_kernel(Allocator &allocator,
                                   BatchNormPerDeviceState &per_device_state) {
  allocator.deallocate(per_device_state.runningMean);
  checkCUDNN(cudnnDestroyTensorDescriptor(per_device_state.inputTensor));
  checkCUDNN(cudnnDestroyTensorDescriptor(per_device_state.outputTensor));
  checkCUDNN(cudnnDestroyTensorDescriptor(per_device_state.biasTensor));
}

} // namespace FlexFlow
