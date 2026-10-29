#ifndef _FLEXFLOW_OP_ATTRS_OPS_OP_ATTRS_INPUT_H
#define _FLEXFLOW_OP_ATTRS_OPS_OP_ATTRS_INPUT_H

#include "op-attrs/ops/input_attrs.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/standard_operator_task_group.h"
#include "op-attrs/tensor_shape.dtg.h"

namespace FlexFlow {

TensorShape input_get_output_shape(InputAttrs const &);
ParallelTensorDimDegrees
    input_get_output_parallel_dim_degrees(InputAttrs const &);
ParallelTensorShape input_get_output_parallel_shape(InputAttrs const &);

StandardOperatorTaskGroup input_get_task_group(InputAttrs const &);

} // namespace FlexFlow

#endif
