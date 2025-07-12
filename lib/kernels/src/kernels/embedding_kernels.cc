#include "kernels/embedding_kernels.h"
#include "kernels/embedding_kernels_cpu.h"
#include "kernels/embedding_kernels_gpu.h"

namespace FlexFlow::Kernels::Embedding {

void forward_kernel(device_stream_t const &stream,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output,
                    GenericTensorAccessorR const &weight,
                    DataType input_data_type,
                    DataType output_data_type,
                    std::optional<AggregateOp> aggr,
                    int in_dim,
                    int out_dim,
                    int batch_size) {
  if (stream.is_gpu()) {
    gpu_forward_kernel(
        /*stream=*/stream.require_gpu(),
        /*input=*/input,
        /*output=*/output,
        /*weight=*/weight,
        /*input_data_type=*/input_data_type,
        /*output_data_type=*/output_data_type,
        /*aggr=*/aggr,
        /*in_dim=*/in_dim,
        /*out_dim=*/out_dim,
        /*batch_size=*/batch_size);
  } else {
    ASSERT(stream.is_cpu());
    cpu_forward_kernel(
        /*input=*/input,
        /*output=*/output,
        /*weight=*/weight,
        /*input_data_type=*/input_data_type,
        /*output_data_type=*/output_data_type,
        /*aggr=*/aggr,
        /*in_dim=*/in_dim,
        /*out_dim=*/out_dim,
        /*batch_size=*/batch_size);
  }
}

void backward_kernel(device_stream_t const &stream,
                     GenericTensorAccessorR const &output,
                     GenericTensorAccessorR const &input,
                     GenericTensorAccessorW const &weight_grad,
                     DataType output_data_type,
                     DataType input_data_type,
                     std::optional<AggregateOp> aggr,
                     int in_dim,
                     int out_dim,
                     int batch_size) {
  if (stream.is_gpu()) {
    gpu_backward_kernel(
        /*stream=*/stream.require_gpu(),
        /*output=*/output,
        /*input=*/input,
        /*weight_grad=*/weight_grad,
        /*output_data_type=*/output_data_type,
        /*input_data_type=*/input_data_type,
        /*aggr=*/aggr,
        /*in_dim=*/in_dim,
        /*out_dim=*/out_dim,
        /*batch_size=*/batch_size);
  } else {
    ASSERT(stream.is_cpu());
    cpu_backward_kernel(
        /*output=*/output,
        /*input=*/input,
        /*weight_grad=*/weight_grad,
        /*output_data_type=*/output_data_type,
        /*input_data_type=*/input_data_type,
        /*aggr=*/aggr,
        /*in_dim=*/in_dim,
        /*out_dim=*/out_dim,
        /*batch_size=*/batch_size);
  }
}

} // namespace FlexFlow::Kernels::Embedding
