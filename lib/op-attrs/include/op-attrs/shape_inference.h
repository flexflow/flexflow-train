#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_SHAPE_INFERENCE_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_SHAPE_INFERENCE_H

#include "op-attrs/computation_graph_op_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/pcg_operator_attrs.dtg.h"
#include "utils/singular_or_variadic.dtg.h"
#include "op-attrs/tensor_slot_name.dtg.h"
#include <vector>

namespace FlexFlow {

std::unordered_map<TensorSlotName, SingularOrVariadic<TensorShape>>
    get_output_shapes(ComputationGraphOpAttrs const &,
                      std::unordered_map<TensorSlotName, SingularOrVariadic<TensorShape>> const &input_shapes);

std::unordered_map<TensorSlotName, SingularOrVariadic<TensorShape>>
    get_weight_shapes(ComputationGraphOpAttrs const &,
                      std::unordered_map<TensorSlotName, SingularOrVariadic<TensorShape>> const &input_shapes);

std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>>
    get_output_shapes(PCGOperatorAttrs const &,
                      std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> const &input_shapes);

std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>>
    get_weight_shapes(PCGOperatorAttrs const &,
                      std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>> const &input_shapes);

} // namespace FlexFlow

#endif
