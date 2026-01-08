#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_REDUCE_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_REDUCE_H

#include "op-attrs/ops/reduce_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"

namespace FlexFlow {

ParallelTensorShape get_output_shape(ReduceAttrs const &,
                                     ParallelTensorShape const &input_shape);

} // namespace FlexFlow

#endif
