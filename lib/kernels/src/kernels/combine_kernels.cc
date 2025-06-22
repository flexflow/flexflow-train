#include "kernels/combine_kernels.h"
#include "kernels/combine_kernels_gpu.h"
#include "kernels/combine_kernels_cpu.h"

namespace FlexFlow::Kernels::Combine {

void forward_kernel(device_stream_t const &stream,
                    GenericTensorAccessorR const &input_accessor,
                    GenericTensorAccessorW const &output_accessor) {
  if (stream.is_gpu()) {
    gpu_forward_kernel(stream.require_gpu(), input_accessor, output_accessor);
  } else {
    ASSERT(stream.is_cpu());
    cpu_forward_kernel(input_accessor, output_accessor);
  }
}

void backward_kernel(device_stream_t const &stream,
                    GenericTensorAccessorR const &output_grad,
                    GenericTensorAccessorW const &input_grad) {
  if (stream.is_gpu()) {
    gpu_backward_kernel(stream.require_gpu(), output_grad, input_grad);
  } else {
    ASSERT(stream.is_cpu());
    cpu_backward_kernel(output_grad, input_grad);
  }
}

} // namespace FlexFlow
