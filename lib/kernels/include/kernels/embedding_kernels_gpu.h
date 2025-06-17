#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_EMBEDDING_KERNELS_GPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_EMBEDDING_KERNELS_GPU_H

#include "kernels/accessor.h"
#include "kernels/device.h"
#include "op-attrs/ops/embedding.h"

namespace FlexFlow::Kernels::Embedding {

void gpu_forward_kernel(ffStream_t stream,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output,
                    GenericTensorAccessorR const &weight,
                    DataType input_data_type,
                    DataType output_data_type,
                    std::optional<AggregateOp> aggr,
                    int in_dim,
                    int out_dim,
                    int batch_size);
void gpu_backward_kernel(ffStream_t stream,
                     GenericTensorAccessorR const &output,
                     GenericTensorAccessorR const &input,
                     GenericTensorAccessorW const &weight_grad,
                     DataType output_data_type,
                     DataType input_data_type,
                     std::optional<AggregateOp> aggr,
                     int in_dim,
                     int out_dim,
                     int batch_size);

} // namespace FlexFlow

#endif
