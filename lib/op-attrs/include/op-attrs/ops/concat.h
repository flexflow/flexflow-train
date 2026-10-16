#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_CONCAT_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_CONCAT_H

#include "op-attrs/ops/concat_attrs.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/standard_operator_task_group.h"
#include "op-attrs/tensor_shape.dtg.h"

namespace FlexFlow {

std::vector<TensorSlotName> concat_get_input_slot_names(ConcatAttrs const &);

TensorShape concat_get_output_shape(ConcatAttrs const &,
                                    std::vector<TensorShape> const &);

ParallelTensorDimDegrees concat_get_output_parallel_dim_degrees(
    ConcatAttrs const &, std::vector<ParallelTensorDimDegrees> const &);

ParallelTensorShape
    concat_get_output_parallel_shape(ConcatAttrs const &,
                                     std::vector<ParallelTensorShape> const &);

StandardOperatorTaskGroup concat_get_task_group(
    ConcatAttrs const &attrs,
    std::vector<ParallelTensorDimDegrees> const &inputs_degrees);

} // namespace FlexFlow

#endif
