#include "kernels/transpose_kernels.h"
#include "kernels/transpose_kernels_cpu.h"
#include "kernels/transpose_kernels_gpu.h"

namespace FlexFlow::Kernels::Transpose {

void forward_kernel(device_stream_t const &stream,
                    TransposeAttrs const &attrs,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output) {
  if (stream.is_gpu()) {
    gpu_forward_kernel(
                       /*stream=*/stream.require_gpu(),
                       /*attrs=*/attrs,
                       /*input=*/input,
                       /*output=*/output);
  } else {
    ASSERT(stream.is_cpu()); 
    cpu_forward_kernel(
                       /*attrs=*/attrs,
                       /*input=*/input,
                       /*output=*/output);
  }
}

void backward_kernel(device_stream_t const &stream,
                     TransposeAttrs const &attrs,
                     GenericTensorAccessorR const &out_grad,
                     GenericTensorAccessorW const &in_grad) {
  if (stream.is_gpu()) {
    gpu_backward_kernel(
                        /*stream=*/stream.require_gpu(),
                        /*attrs=*/attrs,
                        /*out_grad=*/out_grad,
                        /*in_grad=*/in_grad);
  } else {
    ASSERT(stream.is_cpu());
    cpu_forward_kernel(
                       /*attrs=*/attrs,
                       /*out_grad=*/out_grad,
                       /*in_grad=*/in_grad);
  }
}


} // namespace FlexFlow
