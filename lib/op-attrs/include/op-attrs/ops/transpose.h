#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_TRANSPOSE_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_TRANSPOSE_H

#include "op-attrs/operator_task_space.dtg.h"
#include "op-attrs/ops/transpose_attrs.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/shard_signature_instance.h"
#include "op-attrs/standard_operator_task_group.h"
#include "op-attrs/tensor_role.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"

namespace FlexFlow {

std::map<TensorSlotName, TensorRole>
    transpose_get_tensor_roles(TransposeAttrs const &);

TensorShape transpose_get_output_shape(TransposeAttrs const &,
                                       TensorShape const &);

ParallelTensorDimDegrees
    transpose_get_output_parallel_dim_degrees(TransposeAttrs const &,
                                              ParallelTensorDimDegrees const &);

ParallelTensorShape
    transpose_get_output_parallel_shape(TransposeAttrs const &,
                                        ParallelTensorShape const &);

StandardOperatorTaskGroup
    transpose_get_task_group(TransposeAttrs const &,
                             ParallelTensorDimDegrees const &);

ShardSignatureInstance transpose_get_shard_signature_instance(
    TransposeAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees);

OperatorTaskSpace transpose_get_operator_task_space(
    TransposeAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees);

OperatorSpaceToParallelTensorSpaceMapping
    transpose_get_operator_to_input_mapping(
        TransposeAttrs const &attrs,
        ParallelTensorDimDegrees const &input_degrees);

OperatorSpaceToParallelTensorSpaceMapping
    transpose_get_operator_to_output_mapping(
        TransposeAttrs const &attrs,
        ParallelTensorDimDegrees const &input_degrees);

} // namespace FlexFlow

#endif
