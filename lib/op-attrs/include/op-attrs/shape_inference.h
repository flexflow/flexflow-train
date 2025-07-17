#ifndef _FLEXFLOW_INCLUDE_OP_ATTRS_SHAPE_INFERENCE_H
#define _FLEXFLOW_INCLUDE_OP_ATTRS_SHAPE_INFERENCE_H

#include "op-attrs/computation_graph_op_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/pcg_operator_attrs.dtg.h"
#include <vector>

namespace FlexFlow {

std::vector<TensorShape>
    get_output_shapes(ComputationGraphOpAttrs const &,
                      std::vector<TensorShape> const &input_shapes);

std::vector<TensorShape>
    get_weight_shapes(ComputationGraphOpAttrs const &,
                      std::vector<TensorShape> const &input_shapes);

std::vector<ParallelTensorShape>
    get_output_shapes(PCGOperatorAttrs const &,
                      std::vector<ParallelTensorShape> const &input_shapes);

std::vector<ParallelTensorShape>
    get_weight_shapes(PCGOperatorAttrs const &,
                      std::vector<ParallelTensorShape> const &input_shapes);

} // namespace FlexFlow

#endif
