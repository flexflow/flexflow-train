#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_RESHAPE_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_RESHAPE_H

#include "op-attrs/ops/reshape_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"

namespace FlexFlow {

TensorShape get_output_shape(ReshapeAttrs const &attrs,
                             TensorShape const &input_shape);
ParallelTensorShape get_output_shape(ReshapeAttrs const &attrs,
                                     ParallelTensorShape const &input_shape);

} // namespace FlexFlow

#endif
