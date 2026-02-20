#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_SPLIT_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_SPLIT_H

#include "op-attrs/ops/split_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"
#include <vector>

namespace FlexFlow {

std::vector<TensorShape> get_output_shapes(SplitAttrs const &,
                                           TensorShape const &);
std::vector<ParallelTensorShape>
    get_output_shapes(SplitAttrs const &attrs,
                      ParallelTensorShape const &input_shape);

} // namespace FlexFlow

#endif
