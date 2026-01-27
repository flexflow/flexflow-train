#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_SOFTMAX_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_SOFTMAX_H

#include "op-attrs/ops/softmax_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"
#include <tl/expected.hpp>

namespace FlexFlow {

tl::expected<TensorShape, std::string>
    get_output_shape(SoftmaxAttrs const &attrs, TensorShape const &input_shape);
tl::expected<ParallelTensorShape, std::string>
    get_output_shape(SoftmaxAttrs const &attrs,
                     ParallelTensorShape const &input_shape);

} // namespace FlexFlow

#endif
