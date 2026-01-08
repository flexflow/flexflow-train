#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_GET_OPERATOR_TASK_SPACE_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_GET_OPERATOR_TASK_SPACE_H

#include "op-attrs/computation_graph_op_attrs.dtg.h"
#include "op-attrs/operator_task_space.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/tensor_slot_name.dtg.h"

namespace FlexFlow {

OperatorTaskSpace get_operator_task_space(
    ComputationGraphOpAttrs const &attrs,
    std::unordered_map<TensorSlotName, ParallelTensorDimDegrees> const
        &inputs_degrees);

} // namespace FlexFlow

#endif
