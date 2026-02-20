#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_NOOP_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_NOOP_H

#include "op-attrs/ops/noop_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"

namespace FlexFlow {

TensorShape get_output_shape(NoopAttrs const &, TensorShape const &);
ParallelTensorShape get_output_shape(NoopAttrs const &,
                                     ParallelTensorShape const &);

} // namespace FlexFlow

#endif
