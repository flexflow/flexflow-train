#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_WEIGHT_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_WEIGHT_H

#include "op-attrs/ops/weight_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "utils/record_formatter.h"

namespace FlexFlow {

RecordFormatter as_dot(WeightAttrs const &);

TensorShape get_output_shape(WeightAttrs const &);
ParallelTensorShape get_output_parallel_tensor_shape(WeightAttrs const &);

} // namespace FlexFlow

#endif
