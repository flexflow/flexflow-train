#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_FLAT_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_FLAT_H

#include "op-attrs/ops/flat_attrs.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"

namespace FlexFlow {

TensorShape get_output_shape(FlatAttrs const &, TensorShape const &);
ParallelTensorDimDegrees
    get_output_parallel_dim_degrees(FlatAttrs const &,
                                    ParallelTensorDimDegrees const &);
ParallelTensorShape get_output_shape(FlatAttrs const &,
                                     ParallelTensorShape const &);

} // namespace FlexFlow

#endif
