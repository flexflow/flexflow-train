#ifndef _FLEXFLOW_OPS_KERNELS_EMBEDDING_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_EMBEDDING_KERNELS_H

#include "kernels/accessor.h"
#include "kernels/device_stream_t.dtg.h"
#include "op-attrs/ops/embedding_attrs.dtg.h"

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
                    int batch_size);

void backward_kernel(device_stream_t const &stream,
                     GenericTensorAccessorR const &output,
                     GenericTensorAccessorR const &input,
                     GenericTensorAccessorW const &weight_grad,
                     DataType output_data_type,
                     DataType input_data_type,
                     std::optional<AggregateOp> aggr,
                     int in_dim,
                     int out_dim,
                     int batch_size);

} // namespace FlexFlow::Kernels::Embedding

#endif // _FLEXFLOW_OPS_KERNELS_EMBEDDING_KERNELS_H
