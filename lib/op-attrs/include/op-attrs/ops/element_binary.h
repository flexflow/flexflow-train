#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_ELEMENT_BINARY_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_ELEMENT_BINARY_H

#include "op-attrs/ops/element_binary_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.h"

namespace FlexFlow {

TensorShape get_output_shape(ElementBinaryAttrs const &,
                             TensorShape const &,
                             TensorShape const &);
ParallelTensorShape get_output_shape(ElementBinaryAttrs const &,
                                     ParallelTensorShape const &,
                                     ParallelTensorShape const &);

} // namespace FlexFlow

#endif
