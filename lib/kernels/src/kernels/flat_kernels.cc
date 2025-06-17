#include "kernels/flat_kernels.h"
#include "kernels/flat_kernels_cpu.h"
#include "kernels/flat_kernels_gpu.h"

namespace FlexFlow::Kernels::Flat {

void forward_kernel(device_stream_t const &stream,
                    GenericTensorAccessorR const &input,
                    float *output_ptr) {
  if (stream.is_gpu()) {
    gpu_forward_kernel(
                   /*stream=*/stream.require_gpu(),
                   /*input=*/input,
                   /*output_ptr=*/output_ptr);
  } else {
    ASSERT(stream.is_cpu());
    cpu_forward_kernel(
                       /*input=*/input,
                       /*output_ptr=*/output_ptr);
  }
}

void backward_kernel(device_stream_t const &stream,
                     GenericTensorAccessorR const &input,
                     float const *output_grad_ptr,
                     float *input_grad_ptr) {
  if (stream.is_gpu()) {
    gpu_backward_kernel(
                        /*stream=*/stream.require_gpu(),
                        /*input=*/input,
                        /*output_grad_ptr=*/output_grad_ptr,
                        /*input_grad_ptr=*/input_grad_ptr);
  } else {
    ASSERT(stream.is_cpu());
    cpu_backward_kernel(
                        /*input=*/input,
                        /*output_grad_ptr=*/output_grad_ptr,
                        /*input_grad_ptr=*/input_grad_ptr);
                        
                        
  }
  
}


} // namespace FlexFlow
