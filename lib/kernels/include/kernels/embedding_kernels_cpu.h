#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_EMBEDDING_KERNELS_CPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_EMBEDDING_KERNELS_CPU_H

#include "kernels/accessor.h"
#include "op-attrs/ops/embedding_attrs.dtg.h"

namespace FlexFlow::Kernels::Embedding {

void cpu_forward_kernel(GenericTensorAccessorR const &input,
                        GenericTensorAccessorW const &output,
                        GenericTensorAccessorR const &weight,
                        DataType input_data_type,
                        DataType output_data_type,
                        std::optional<AggregateOp> aggr,
                        int in_dim,
                        int out_dim,
                        int batch_size);

void cpu_backward_kernel(GenericTensorAccessorR const &output,
                         GenericTensorAccessorR const &input,
                         GenericTensorAccessorW const &weight_grad,
                         DataType output_data_type,
                         DataType input_data_type,
                         std::optional<AggregateOp> aggr,
                         int in_dim,
                         int out_dim,
                         int batch_size);

} // namespace FlexFlow::Kernels::Embedding

#endif
