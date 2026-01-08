#ifndef _FLEXFLOW_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_REVERSE_H
#define _FLEXFLOW_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_REVERSE_H

#include "op-attrs/ops/reverse_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"

namespace FlexFlow {

TensorShape get_output_shape(ReverseAttrs const &, TensorShape const &);
ParallelTensorShape get_output_shape(ReverseAttrs const &attrs,
                                     ParallelTensorShape const &input_shape);

} // namespace FlexFlow

#endif
