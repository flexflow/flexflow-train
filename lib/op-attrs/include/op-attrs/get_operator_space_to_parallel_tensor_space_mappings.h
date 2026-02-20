#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_GET_OPERATOR_SPACE_TO_PARALLEL_TENSOR_SPACE_MAPPINGS_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_GET_OPERATOR_SPACE_TO_PARALLEL_TENSOR_SPACE_MAPPINGS_H

#include "op-attrs/computation_graph_op_attrs.dtg.h"
#include "op-attrs/incoming_tensor_role.dtg.h"
#include "op-attrs/num_ptensor_parallel_dims_t.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_mapping.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/tensor_role.dtg.h"
#include "op-attrs/tensor_slot_name.dtg.h"
#include <vector>

namespace FlexFlow {

std::unordered_map<TensorSlotName, OperatorSpaceToParallelTensorSpaceMapping>
    get_operator_to_incoming_mappings(
        ComputationGraphOpAttrs const &attrs,
        std::unordered_map<TensorSlotName, ParallelTensorDimDegrees> const
            &inputs_degrees);

std::unordered_map<TensorSlotName, OperatorSpaceToParallelTensorSpaceMapping>
    get_operator_to_incoming_mappings_for_role(
        ComputationGraphOpAttrs const &attrs,
        std::unordered_map<TensorSlotName, ParallelTensorDimDegrees> const
            &inputs_degrees,
        IncomingTensorRole role);

std::unordered_map<TensorSlotName, OperatorSpaceToParallelTensorSpaceMapping>
    get_operator_to_input_mappings(
        ComputationGraphOpAttrs const &attrs,
        std::unordered_map<TensorSlotName, ParallelTensorDimDegrees> const
            &inputs_degrees);

std::unordered_map<TensorSlotName, OperatorSpaceToParallelTensorSpaceMapping>
    get_operator_to_weight_mappings(
        ComputationGraphOpAttrs const &attrs,
        std::unordered_map<TensorSlotName, ParallelTensorDimDegrees> const
            &inputs_degrees);

std::unordered_map<TensorSlotName, OperatorSpaceToParallelTensorSpaceMapping>
    get_operator_to_output_mappings(
        ComputationGraphOpAttrs const &attrs,
        std::unordered_map<TensorSlotName, ParallelTensorDimDegrees> const
            &inputs_degrees);

std::unordered_map<TensorSlotName, OperatorSpaceToParallelTensorSpaceMapping>
    get_operator_to_ptensor_mappings_for_role(
        ComputationGraphOpAttrs const &attrs,
        std::unordered_map<TensorSlotName, ParallelTensorDimDegrees> const
            &inputs_degrees,
        TensorRole role);

std::unordered_map<TensorSlotName, OperatorSpaceToParallelTensorSpaceMapping>
    get_operator_to_ptensor_mappings(
        ComputationGraphOpAttrs const &attrs,
        std::unordered_map<TensorSlotName, ParallelTensorDimDegrees> const
            &inputs_degrees);

} // namespace FlexFlow

#endif
