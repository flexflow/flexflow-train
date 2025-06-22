#include "kernels/pool_2d_kernels.h"
#include "kernels/pool_2d_kernels_cpu.h"
#include "kernels/pool_2d_kernels_gpu.h"

namespace FlexFlow::Kernels::Pool2D {

std::optional<Pool2DPerDeviceState> init_kernel(DeviceType device_type, 
                                            PerDeviceFFHandle handle,
                                 std::optional<Activation> activation,
                                 int input_w,
                                 int input_h,
                                 int input_c,
                                 int input_n,
                                 int output_w,
                                 int output_h,
                                 int output_c,
                                 int output_n,
                                 int pad_h,
                                 int pad_w,
                                 int kernel_h,
                                 int kernel_w,
                                 int stride_h,
                                 int stride_w,
                                 PoolOp pool_type) {
  if (device_type == DeviceType::GPU) {
    return gpu_init_kernel(
                           /*handle=*/handle,
                           /*activation=*/activation,
                           /*input_w=*/input_w,
                           /*input_h=*/input_h,
                           /*input_c=*/input_c,
                           /*input_n=*/input_n,
                           /*output_w=*/output_w,
                           /*output_h=*/output_h,
                           /*output_c=*/output_c,
                           /*output_n=*/output_n,
                           /*pad_h=*/pad_h,
                           /*pad_w=*/pad_w,
                           /*kernel_h=*/kernel_h,
                           /*kernel_w=*/kernel_w,
                           /*stride_h=*/stride_h,
                           /*stride_w=*/stride_w,
                           /*pool_type=*/pool_type);
  } else {
    ASSERT(device_type == DeviceType::CPU);
    return std::nullopt;
  }
  
}

void forward_kernel(device_stream_t const &stream,
                    std::optional<Pool2DPerDeviceState> const &per_device_state,
                    void const *input_ptr,
                    void *output_ptr) {
  if (stream.is_gpu()) {
    gpu_forward_kernel(
                       /*stream=*/stream.require_gpu(),
                       /*per_device_state=*/per_device_state.value(),
                       /*input_ptr=*/input_ptr,
                       /*output_ptr=*/output_ptr);
  } else {
    ASSERT(stream.is_cpu());
    cpu_forward_kernel(
                       /*input_ptr=*/input_ptr,
                       /*output_ptr=*/output_ptr);
  }
}

void backward_kernel(device_stream_t const &stream,
                     std::optional<Pool2DPerDeviceState> const &per_device_state,
                     void const *output_ptr,
                     void const *output_grad_ptr,
                     void const *input_ptr,
                     void *input_grad_ptr) {
  if (stream.is_gpu()) {
    gpu_backward_kernel(
                        /*stream=*/stream.require_gpu(),
                        /*per_device_state=*/per_device_state.value(),
                        /*output_ptr=*/output_ptr,
                        /*output_grad_ptr=*/output_grad_ptr,
                        /*input_ptr=*/input_ptr,
                        /*input_grad_ptr=*/input_grad_ptr);
  } else {
    ASSERT(stream.is_cpu());
    cpu_backward_kernel(
                        /*output_ptr=*/output_ptr,
                        /*output_grad_ptr=*/output_grad_ptr,
                        /*input_ptr=*/input_ptr,
                        /*input_grad_ptr=*/input_grad_ptr);
  }
}

void cleanup_kernel(DeviceType device_type, 
                    std::optional<Pool2DPerDeviceState> &per_device_state) {
  if (device_type == DeviceType::GPU) {
    gpu_cleanup_kernel(per_device_state.value()); 
  } else {
    ASSERT(device_type == DeviceType::CPU); 
    ASSERT(per_device_state == std::nullopt);
  }
}

} // namespace FlexFlow
