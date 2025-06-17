#include "kernels/reverse_kernels.h"
#include "kernels/reverse_kernels_cpu.h"
#include "kernels/reverse_kernels_gpu.h"

namespace FlexFlow::Kernels::Reverse {

void forward_kernel(device_stream_t const &stream,
                    GenericTensorAccessorR const &input_accessor,
                    GenericTensorAccessorW &output_accessor,
                    ReverseAttrs const &attrs) {
  if (stream.is_gpu()) {
    gpu_forward_kernel(stream.require_gpu(), input_accessor, output_accessor, attrs);
  } else {
    ASSERT(stream.is_cpu());
    cpu_forward_kernel(input_accessor, output_accessor, attrs);
  }
}

void backward_kernel(device_stream_t const &stream,
                    GenericTensorAccessorR const &output_accessor,
                    GenericTensorAccessorW &input_accessor,
                    ReverseAttrs const &attrs) {
  if (stream.is_gpu()) {
    gpu_backward_kernel(stream.require_gpu(), output_accessor, input_accessor, attrs);
  } else {
    ASSERT(stream.is_cpu());
    cpu_backward_kernel(output_accessor, input_accessor, attrs);
  }
}


} // namespace FlexFlow
