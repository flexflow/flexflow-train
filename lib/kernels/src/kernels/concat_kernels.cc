#include "kernels/concat_kernels.h"
#include "kernels/concat_kernels_cpu.h"
#include "kernels/concat_kernels_gpu.h"

namespace FlexFlow::Kernels::Concat {

void forward_kernel(device_stream_t const &stream,
                    GenericTensorAccessorW const &output,
                    std::vector<GenericTensorAccessorR> const &inputs,
                    ff_dim_t axis) {
  if (stream.is_gpu()) {
    gpu_forward_kernel(
        /*stream=*/stream.require_gpu(),
        /*output=*/output,
        /*inputs=*/inputs,
        /*axis=*/axis);
  } else {
    ASSERT(stream.is_cpu());
    cpu_forward_kernel(
        /*output=*/output,
        /*inputs=*/inputs,
        /*axis=*/axis);
  }
}

void backward_kernel(device_stream_t const &stream,
                     GenericTensorAccessorR const &output_grad,
                     std::vector<GenericTensorAccessorW> const &input_grads,
                     ff_dim_t axis) {
  if (stream.is_gpu()) {
    gpu_backward_kernel(
        /*stream=*/stream.require_gpu(),
        /*output_grad=*/output_grad,
        /*input_grads=*/input_grads,
        /*axis=*/axis);
  } else {
    ASSERT(stream.is_cpu());
    cpu_backward_kernel(
        /*output_grad=*/output_grad,
        /*input_grads=*/input_grads,
        /*axis=*/axis);
  }
}

} // namespace FlexFlow::Kernels::Concat
