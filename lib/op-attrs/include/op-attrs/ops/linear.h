#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_LINEAR_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_LINEAR_H

#include "op-attrs/incoming_tensor_role.dtg.h"
#include "op-attrs/initializer_attrs.dtg.h"
#include "op-attrs/num_ptensor_parallel_dims_t.h"
#include "op-attrs/ops/linear_attrs.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/standard_operator_task_group.h"
#include "op-attrs/tensor_role.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "op-attrs/tensor_slot_name.dtg.h"

namespace FlexFlow {

std::map<TensorSlotName, IncomingTensorRole>
    get_linear_incoming_tensor_roles(LinearAttrs const &);

std::set<TensorSlotName> linear_get_slots(LinearAttrs const &);

std::map<TensorSlotName, TensorRole>
    linear_get_tensor_roles(LinearAttrs const &);

TensorShape linear_get_projection_shape(LinearAttrs const &attrs,
                                        TensorShape const &input);
TensorShape linear_get_bias_shape(LinearAttrs const &attrs,
                                  TensorShape const &input);
TensorShape linear_get_output_shape(LinearAttrs const &attrs,
                                    TensorShape const &input);

std::map<TensorSlotName, TensorShape>
    linear_get_weight_shapes(LinearAttrs const &attrs,
                             TensorShape const &input_shape);

ParallelTensorDimDegrees linear_get_projection_parallel_dim_degrees(
    LinearAttrs const &attrs, ParallelTensorDimDegrees const &input);
ParallelTensorDimDegrees
    linear_get_bias_parallel_dim_degrees(LinearAttrs const &attrs,
                                         ParallelTensorDimDegrees const &input);
ParallelTensorDimDegrees linear_get_output_parallel_dim_degrees(
    LinearAttrs const &attrs, ParallelTensorDimDegrees const &input);

ParallelTensorShape
    linear_get_projection_parallel_shape(LinearAttrs const &attrs,
                                         ParallelTensorShape const &input);
ParallelTensorShape
    linear_get_bias_parallel_shape(LinearAttrs const &attrs,
                                   ParallelTensorShape const &input);
ParallelTensorShape
    linear_get_output_parallel_shape(LinearAttrs const &attrs,
                                     ParallelTensorShape const &input);

std::map<TensorSlotName, ParallelTensorShape>
    linear_get_weight_parallel_shapes(LinearAttrs const &attrs,
                                      ParallelTensorShape const &input_shape);

std::map<TensorSlotName, InitializerAttrs> linear_get_initializers(
    LinearAttrs const &,
    TensorShape const &input_shape,
    std::optional<InitializerAttrs> const &projection_initializer =
        std::nullopt,
    std::optional<InitializerAttrs> const &kernel_initializer = std::nullopt);

StandardOperatorTaskGroup
    linear_get_task_group(LinearAttrs const &attrs,
                          ParallelTensorDimDegrees const &input_degrees);

ShardSignatureInstance linear_get_shard_signature_instance(
    LinearAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees);

OperatorTaskSpace linear_get_operator_task_space(
    LinearAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees);

OperatorSpaceToParallelTensorSpaceMapping linear_get_operator_to_input_mapping(
    LinearAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees);

OperatorSpaceToParallelTensorSpaceMapping
    linear_get_operator_to_projection_mapping(
        LinearAttrs const &attrs,
        ParallelTensorDimDegrees const &input_degrees);

OperatorSpaceToParallelTensorSpaceMapping linear_get_operator_to_bias_mapping(
    LinearAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees);

OperatorSpaceToParallelTensorSpaceMapping linear_get_operator_to_output_mapping(
    LinearAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees);

} // namespace FlexFlow

#endif
