#include "kernels/layer_norm_kernels.h"
#include "kernels/layer_norm_kernels_cpu.h"
#include "kernels/layer_norm_kernels_gpu.h"

namespace FlexFlow::Kernels::LayerNorm {

std::optional<LayerNormPerDeviceState> init_kernel(DeviceType device_type,
                                                   PerDeviceFFHandle const &handle,
                                    Allocator &allocator,
                                    bool elementwise_affine,
                                    int64_t effective_batch_size,
                                    int64_t effective_num_elements,
                                    float eps) {
  if (device_type == DeviceType::GPU) {
    return gpu_init_kernel(
                           /*handle=*/handle,
                           /*allocator=*/allocator,
                           /*elementwise_affine=*/elementwise_affine,
                           /*effective_batch_size=*/effective_batch_size,
                           /*effective_num_elements=*/effective_num_elements,
                           /*eps=*/eps);
  } else {
    ASSERT(device_type == DeviceType::CPU);
    return std::nullopt;
  }
}

void forward_kernel(device_stream_t const &stream,
                    std::optional<LayerNormPerDeviceState> const &per_device_state,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output,
                    GenericTensorAccessorW const &gamma,
                    GenericTensorAccessorW const &beta) {
  if (stream.is_gpu()) {
    gpu_forward_kernel(
                       /*stream=*/stream.require_gpu(),
                       /*per_device_state=*/per_device_state.value(),
                       /*input=*/input,
                       /*output=*/output,
                       /*gamma=*/gamma,
                       /*beta=*/beta);
  } else {
    ASSERT(stream.is_cpu());
    ASSERT(per_device_state == std::nullopt);
    cpu_forward_kernel(
                       /*input=*/input,
                       /*output=*/output,
                       /*gamma=*/gamma,
                       /*beta=*/beta);
  }
  
}

void backward_kernel(device_stream_t const &stream,
                     std::optional<LayerNormPerDeviceState> const &per_device_state,
                     GenericTensorAccessorR const &output_grad,
                     GenericTensorAccessorR const &input,
                     GenericTensorAccessorW const &input_grad,
                     GenericTensorAccessorR const &gamma,
                     GenericTensorAccessorW const &gamma_grad,
                     GenericTensorAccessorW const &beta_grad) {
  if (stream.is_gpu()) {
    gpu_backward_kernel(
                        /*stream=*/stream.require_gpu(),
                        /*per_device_state=*/per_device_state.value(),
                        /*output_grad=*/output_grad,
                        /*input=*/input,
                        /*input_grad=*/input_grad,
                        /*gamma=*/gamma,
                        /*gamma_grad=*/gamma_grad,
                        /*beta_grad=*/beta_grad);
  } else {
    ASSERT(stream.is_cpu());
    ASSERT(per_device_state == std::nullopt);
    cpu_backward_kernel(
                        /*output_grad=*/output_grad,
                        /*input=*/input,
                        /*input_grad=*/input_grad,
                        /*gamma=*/gamma,
                        /*gamma_grad=*/gamma_grad,
                        /*beta_grad=*/beta_grad);
  }
}

void cleanup_kernel(DeviceType device_type,
                    std::optional<LayerNormPerDeviceState> const &per_device_state) {
  if (device_type == DeviceType::GPU) {
    gpu_cleanup_kernel(per_device_state.value());
  } else {
    ASSERT(device_type == DeviceType::CPU);
    ASSERT(per_device_state == std::nullopt);
  }
}


} // namespace FlexFlow
