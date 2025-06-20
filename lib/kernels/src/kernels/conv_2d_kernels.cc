#include "kernels/conv_2d_kernels.h"
#include "kernels/conv_2d_kernels_cpu.h"
#include "kernels/conv_2d_kernels_gpu.h"

namespace FlexFlow::Kernels::Conv2D {

std::optional<Conv2DPerDeviceState> init_kernel(DeviceType device_type,
                                                PerDeviceFFHandle handle,
                                 std::optional<Activation> activation,
                                 int kernel_h,
                                 int kernel_w,
                                 int groups,
                                 int padding_h,
                                 int padding_w,
                                 int stride_h,
                                 int stride_w,
                                 GenericTensorAccessorW const &input,
                                 GenericTensorAccessorW const &output,
                                 float const *filter_ptr,
                                 float *filter_grad_ptr) {
  if (device_type == DeviceType::GPU) {
    return gpu_init_kernel(
                    /*handle=*/handle,
                    /*activation=*/activation,
                    /*kernel_h=*/kernel_h,
                    /*kernel_w=*/kernel_w,
                    /*groups=*/groups,
                    /*padding_h=*/padding_h,
                    /*padding_w=*/padding_w,
                    /*stride_h=*/stride_h,
                    /*stride_w=*/stride_w,
                    /*input=*/input,
                    /*output=*/output,
                    /*filter_ptr=*/filter_ptr,
                    /*filter_grad_ptr=*/filter_grad_ptr);
  } else {
    ASSERT(device_type == DeviceType::CPU); 
    return std::nullopt;
  }
}

void forward_kernel(device_stream_t const &stream,
                    std::optional<Conv2DPerDeviceState> const &per_device_state,
                    float const *input_ptr,
                    float *output_ptr,
                    float const *filter_ptr,
                    float const *bias_ptr,
                    std::optional<Activation> activation) {
  if (stream.is_gpu()) {
    gpu_forward_kernel(
                       /*stream=*/stream.require_gpu(),
                       /*per_device_state=*/per_device_state.value(),
                       /*input_ptr=*/input_ptr,
                       /*output_ptr=*/output_ptr,
                       /*filter_ptr=*/filter_ptr,
                       /*bias_ptr=*/bias_ptr,
                       /*activation=*/activation);
  } else {
    ASSERT(stream.is_cpu());
    cpu_forward_kernel(
                       /*input_ptr=*/input_ptr,
                       /*output_ptr=*/output_ptr,
                       /*filter_ptr=*/filter_ptr,
                       /*bias_ptr=*/bias_ptr,
                       /*activation=*/activation);
  }
}

void backward_kernel(device_stream_t const &stream,
                     std::optional<Conv2DPerDeviceState> const &per_device_state,
                     float const *output_ptr,
                     float *output_grad_ptr,
                     float const *input_ptr,
                     float *input_grad_ptr,
                     float const *filter_ptr,
                     float *filter_grad_ptr,
                     float *bias_grad_ptr,
                     std::optional<Activation> activation) {
  if (stream.is_gpu()) {
    gpu_backward_kernel(
                        /*stream=*/stream.require_gpu(),
                        /*per_device_state=*/per_device_state.value(),
                        /*output_ptr=*/output_ptr,
                        /*output_grad_ptr=*/output_grad_ptr,
                        /*input_ptr=*/input_ptr,
                        /*input_grad_ptr=*/input_grad_ptr,
                        /*filter_ptr=*/filter_ptr,
                        /*filter_grad_ptr=*/filter_grad_ptr,
                        /*bias_grad_ptr=*/bias_grad_ptr,
                        /*activation=*/activation);
  } else {
    ASSERT(stream.is_cpu()); 
    cpu_backward_kernel(
                        /*output_ptr=*/output_ptr,
                        /*output_grad_ptr=*/output_grad_ptr,
                        /*input_ptr=*/input_ptr,
                        /*input_grad_ptr=*/input_grad_ptr,
                        /*filter_ptr=*/filter_ptr,
                        /*filter_grad_ptr=*/filter_grad_ptr,
                        /*bias_grad_ptr=*/bias_grad_ptr,
                        /*activation=*/activation);
  }
}

void cleanup_kernel(DeviceType device_type,
                    std::optional<Conv2DPerDeviceState> &per_device_state) {
  if (device_type == DeviceType::GPU) {
    gpu_cleanup_kernel(per_device_state.value()); 
  } else {
    ASSERT(device_type == DeviceType::CPU);
    ASSERT(per_device_state == std::nullopt);
  }
}

} // namespace FlexFlow
