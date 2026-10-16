#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_WEIGHT_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_WEIGHT_H

#include "op-attrs/ops/weight_attrs.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/standard_operator_task_group.h"
#include "op-attrs/tensor_shape.dtg.h"

namespace FlexFlow {

TensorShape weight_get_output_shape(WeightAttrs const &);
ParallelTensorDimDegrees
    weight_get_output_parallel_dim_degrees(WeightAttrs const &);
ParallelTensorShape weight_get_output_parallel_shape(WeightAttrs const &);

StandardOperatorTaskGroup weight_get_task_group(WeightAttrs const &);

OperatorTaskSpace weight_get_operator_task_space(WeightAttrs const &);

OperatorSpaceToParallelTensorSpaceMapping
    weight_get_operator_to_output_mapping(WeightAttrs const &);

} // namespace FlexFlow

#endif
