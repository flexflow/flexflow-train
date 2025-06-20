#include "kernels/replicate_kernels.h"
#include "kernels/replicate_kernels_cpu.h"
#include "kernels/replicate_kernels_gpu.h"

namespace FlexFlow::Kernels::Replicate {

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
    cpu_forward_kernel(/*input=*/input,
                       /*output=*/output);
  }
}

void backward_kernel(device_stream_t const &stream,
                     GenericTensorAccessorR const &output,
                     GenericTensorAccessorW const &input,
                     size_t num_replicas) {
  if (stream.is_gpu()) {
    gpu_backward_kernel(
                        /*stream=*/stream.require_gpu(),
                        /*output=*/output,
                        /*input=*/input,
                        /*num_replicas=*/num_replicas);
  } else {
    ASSERT(stream.is_cpu());    
    cpu_backward_kernel(
                        /*output=*/output,
                        /*input=*/input,
                        /*num_replicas=*/num_replicas);
  }
  
}

} // namespace FlexFlow
