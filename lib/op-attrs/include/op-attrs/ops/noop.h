#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_NOOP_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_NOOP_H

#include "op-attrs/ops/noop_attrs.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/standard_operator_task_group.h"
#include "op-attrs/tensor_shape.dtg.h"

namespace FlexFlow {

TensorShape noop_get_output_shape(NoopAttrs const &, TensorShape const &);
ParallelTensorDimDegrees
    noop_get_output_parallel_dim_degrees(NoopAttrs const &,
                                         ParallelTensorDimDegrees const &);
ParallelTensorShape noop_get_output_parallel_shape(NoopAttrs const &,
                                                   ParallelTensorShape const &);

StandardOperatorTaskGroup noop_get_task_group(NoopAttrs const &,
                                              ParallelTensorDimDegrees const &);

} // namespace FlexFlow

#endif
