#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_REDUCE_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_REDUCE_H

#include "op-attrs/ops/reduce_attrs.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/shard_signature_instance.h"
#include "op-attrs/standard_operator_task_group.h"
#include "op-attrs/tensor_shape.dtg.h"

namespace FlexFlow {

TensorShape reduce_get_output_shape(ReduceAttrs const &, TensorShape const &);

ParallelTensorDimDegrees
    reduce_get_output_parallel_dim_degrees(ReduceAttrs const &,
                                           ParallelTensorDimDegrees const &);

ParallelTensorShape
    reduce_get_output_parallel_shape(ReduceAttrs const &,
                                     ParallelTensorShape const &);

StandardOperatorTaskGroup
    reduce_get_task_group(ReduceAttrs const &attrs,
                          ParallelTensorDimDegrees const &input_degrees);

} // namespace FlexFlow

#endif
