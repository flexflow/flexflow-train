#ifndef _FLEXFLOW_REPLICATE_ATTRS_H
#define _FLEXFLOW_REPLICATE_ATTRS_H

#include "op-attrs/ops/core.h"
#include "op-attrs/ops/replicate_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "utils/record_formatter.h"

namespace FlexFlow {

CHECK_VALID_OP_ATTR(ReplicateAttrs);

RecordFormatter as_dot(ReplicateAttrs const &);

ParallelTensorShape get_output_shape(ReplicateAttrs const &attrs,
                                     ParallelTensorShape const &input_shape);

} // namespace FlexFlow

#endif
