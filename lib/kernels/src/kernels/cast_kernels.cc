#include "kernels/cast_kernels.h"
#include "kernels/cast_kernels_cpu.h"
#include "kernels/cast_kernels_gpu.h"

namespace FlexFlow::Kernels::Cast {

void forward_kernel(device_stream_t const &stream,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output) {
  if (stream.is_gpu()) {
    gpu_forward_kernel(
                       /*stream=*/stream.require_gpu(),
                       /*input=*/input,
                       /*output=*/output);
  } else {
    ASSERT(stream.is_cpu());
    cpu_forward_kernel(
                       /*input=*/input,
                       /*output=*/output);
  }
}

void backward_kernel(device_stream_t const &stream,
                     GenericTensorAccessorR const &output_grad,
                     GenericTensorAccessorW const &input_grad) {
  if (stream.is_gpu()) {
    gpu_backward_kernel(
                        /*stream=*/stream.require_gpu(),
                        /*output_grad=*/output_grad,
                        /*input_grad=*/input_grad,
                        
  } else {
    ASSERT(stream.is_cpu());
    cpu_backward_kernel(
                        /*output_grad=*/output_grad,
                        /*input_grad=*/input_grad);
                        
  }
}


} // namespace FlexFlow
