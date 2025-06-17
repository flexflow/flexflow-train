#include "kernels/batch_norm_kernels.h"
#include "kernels/batch_norm_kernels_cpu.h"
#include "kernels/batch_norm_kernels_gpu.h"

namespace FlexFlow::Kernels::BatchNorm {

std::optional<BatchNormPerDeviceState> init_kernel(DeviceType device_type,
                                                   PerDeviceFFHandle const &handle,
                                    Allocator &allocator,
                                    float *runningMean,
                                    int output_n,
                                    int output_c,
                                    int output_h,
                                    int output_w,
                                    bool relu) {
  if (device_type == DeviceType::GPU) {
    return gpu_init_kernel(
                           /*handle=*/handle,
                           /*allocator=*/allocator,
                           /*runningMean=*/runningMean,
                           /*output_n=*/output_n,
                           /*output_c=*/output_c,
                           /*output_h=*/output_h,
                           /*output_w=*/output_w,
                           /*relu=*/relu);
  } else {
    ASSERT(device_type == DeviceType::CPU);
    return std::nullopt;
  }
  
}

void forward_kernel(device_stream_t const &stream,
                    BatchNormPerDeviceState const &per_device_state,
                    float const *input_ptr,
                    float *output_ptr,
                    float const *scale_ptr,
                    float const *bias_ptr) {
  if (stream.is_gpu()) {
    gpu_forward_kernel(
                       /*stream=*/stream.require_gpu(),
                       /*per_device_state=*/per_device_state,
                       /*input_ptr=*/input_ptr,
                       /*output_ptr=*/output_ptr,
                       /*scale_ptr=*/scale_ptr,
                       /*bias_ptr=*/bias_ptr);
  } else {
    ASSERT(stream.is_cpu()); 
    cpu_forward_kernel(
                       /*per_device_state=*/per_device_state,
                       /*input_ptr=*/input_ptr,
                       /*output_ptr=*/output_ptr,
                       /*scale_ptr=*/scale_ptr,
                       /*bias_ptr=*/bias_ptr);
  }
  
}

void backward_kernel(device_stream_t const &stream,
                     BatchNormPerDeviceState const &per_device_state,
                     float const *output_ptr,
                     float *output_grad_ptr,
                     float const *input_ptr,
                     float *input_grad_ptr,
                     float const *scale_ptr,
                     float *scale_grad_ptr,
                     float *bias_grad_ptr,
                     size_t numElements) {
  if (stream.is_gpu()) {
    gpu_backward_kernel(
                        /*stream=*/stream.require_gpu(),
                        /*per_device_state=*/per_device_state,
                        /*output_ptr=*/output_ptr,
                        /*output_grad_ptr=*/output_grad_ptr,
                        /*input_ptr=*/input_ptr,
                        /*input_grad_ptr=*/input_grad_ptr,
                        /*scale_ptr=*/scale_ptr,
                        /*scale_grad_ptr=*/scale_grad_ptr,
                        /*bias_grad_ptr=*/bias_grad_ptr,
                        /*numElements=*/numElements);
  } else {
    ASSERT(stream.is_cpu()); 
    cpu_backward_kernel(
                        /*per_device_state=*/per_device_state,
                        /*output_ptr=*/output_ptr,
                        /*output_grad_ptr=*/output_grad_ptr,
                        /*input_ptr=*/input_ptr,
                        /*input_grad_ptr=*/input_grad_ptr,
                        /*scale_ptr=*/scale_ptr,
                        /*scale_grad_ptr=*/scale_grad_ptr,
                        /*bias_grad_ptr=*/bias_grad_ptr,
                        /*numElements=*/numElements);
  }
  
}

void cleanup_kernel(DeviceType device_type,
                    Allocator &allocator,
                    std::optional<BatchNormPerDeviceState> &per_device_state) {
  if (device_type == DeviceType::GPU) {
    gpu_cleanup_kernel(allocator, per_device_state.value()); 
  } else {
    ASSERT(device_type == DeviceType::CPU);
    ASSERT(per_device_state == std::nullopt);
  }
}

} // namespace FlexFlow
