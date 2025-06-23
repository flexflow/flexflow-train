#include "kernels/reduction_kernels.h"
#include "kernels/reduction_kernels_cpu.h"
#include "kernels/reduction_kernels_gpu.h"

namespace FlexFlow::Kernels::Reduction {

void forward_kernel(device_stream_t const &stream,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output,
                    size_t num_replicas) {
  if (stream.is_gpu()) {
    gpu_forward_kernel(
        /*stream=*/stream.require_gpu(),
        /*input=*/input,
        /*output=*/output,
        /*num_replicas=*/num_replicas);
  } else {
    ASSERT(stream.is_cpu());
    cpu_forward_kernel(
        /*input=*/input,
        /*output=*/output,
        /*num_replicas=*/num_replicas);
  }
}

void backward_kernel(device_stream_t const &stream,
                     GenericTensorAccessorR const &output,
                     GenericTensorAccessorW const &input) {
  if (stream.is_gpu()) {
    gpu_backward_kernel(
        /*stream=*/stream.require_gpu(),
        /*output=*/output,
        /*input=*/input);
  } else {
    ASSERT(stream.is_cpu());
    cpu_backward_kernel(
        /*output=*/output,
        /*input=*/input);
  }
}

} // namespace FlexFlow::Kernels::Reduction
