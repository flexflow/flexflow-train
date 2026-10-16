#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_ELEMENT_BINARY_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_ELEMENT_BINARY_H

#include "op-attrs/ops/element_binary_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/standard_operator_task_group.h"

namespace FlexFlow {

TensorShape element_binary_get_output_shape(ElementBinaryAttrs const &,
                                            TensorShape const &,
                                            TensorShape const &);
ParallelTensorShape
    element_binary_get_output_parallel_shape(ElementBinaryAttrs const &,
                                             ParallelTensorShape const &,
                                             ParallelTensorShape const &);

ParallelTensorDimDegrees element_binary_get_output_parallel_dim_degrees(
    ElementBinaryAttrs const &attrs,
    ParallelTensorDimDegrees const &lhs_input_degrees,
    ParallelTensorDimDegrees const &rhs_input_degrees);

StandardOperatorTaskGroup element_binary_get_task_group(
    ElementBinaryAttrs const &attrs,
    ParallelTensorDimDegrees const &lhs_input_degrees,
    ParallelTensorDimDegrees const &rhs_input_degrees);

ShardSignatureInstance element_binary_get_shard_signature_instance(
    ElementBinaryAttrs const &attrs,
    ParallelTensorDimDegrees const &lhs_input_degrees,
    ParallelTensorDimDegrees const &rhs_input_degrees);

OperatorTaskSpace element_binary_get_operator_task_space(
    ElementBinaryAttrs const &attrs,
    ParallelTensorDimDegrees const &lhs_input_degrees,
    ParallelTensorDimDegrees const &rhs_input_degrees);

OperatorSpaceToParallelTensorSpaceMapping
    element_binary_get_operator_to_lhs_input_mapping(
        ElementBinaryAttrs const &attrs,
        ParallelTensorDimDegrees const &lhs_input_degrees,
        ParallelTensorDimDegrees const &rhs_input_degrees);

OperatorSpaceToParallelTensorSpaceMapping
    element_binary_get_operator_to_rhs_input_mapping(
        ElementBinaryAttrs const &attrs,
        ParallelTensorDimDegrees const &lhs_input_degrees,
        ParallelTensorDimDegrees const &rhs_input_degrees);

OperatorSpaceToParallelTensorSpaceMapping
    element_binary_get_operator_to_output_mapping(
        ElementBinaryAttrs const &attrs,
        ParallelTensorDimDegrees const &lhs_input_degrees,
        ParallelTensorDimDegrees const &rhs_input_degrees);

} // namespace FlexFlow

#endif
