#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_REPLICATE_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_REPLICATE_H

#include "op-attrs/ops/replicate_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "utils/record_formatter.h"

namespace FlexFlow {

RecordFormatter as_dot(ReplicateAttrs const &);

ParallelTensorShape get_output_shape(ReplicateAttrs const &attrs,
                                     ParallelTensorShape const &input_shape);

} // namespace FlexFlow

#endif
