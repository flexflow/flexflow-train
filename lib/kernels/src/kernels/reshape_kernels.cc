#include "kernels/reshape_kernels.h"
#include "kernels/reshape_kernels_cpu.h"
#include "kernels/reshape_kernels_gpu.h"

namespace FlexFlow::Kernels::Reshape {

void forward_kernel(device_stream_t const &stream,
                    DataType data_type,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output) {
  if (stream.is_gpu()) {
    gpu_forward_kernel(
        /*stream=*/stream.require_gpu(),
        /*data_type=*/data_type,
        /*input=*/input,
        /*output=*/output);
  } else {
    ASSERT(stream.is_cpu());
    cpu_forward_kernel(
        /*data_type=*/data_type,
        /*input=*/input,
        /*output=*/output);
  }
}

void backward_kernel(device_stream_t const &stream,
                     DataType data_type,
                     GenericTensorAccessorR const &output,
                     GenericTensorAccessorW const &input) {
  if (stream.is_gpu()) {
    gpu_backward_kernel(
        /*stream=*/stream.require_gpu(),
        /*data_type=*/data_type,
        /*output=*/output,
        /*input=*/input);
  } else {
    ASSERT(stream.is_cpu());
    cpu_backward_kernel(
        /*data_type=*/data_type,
        /*output=*/output,
        /*input=*/input);
  }
}

} // namespace FlexFlow::Kernels::Reshape
