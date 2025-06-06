#ifndef _FLEXFLOW_ELEMENTARY_UNARY_ATTRS_H
#define _FLEXFLOW_ELEMENTARY_UNARY_ATTRS_H

#include "op-attrs/ops/core.h"
#include "op-attrs/ops/element_unary_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"
#include <tl/expected.hpp>

namespace FlexFlow {

ElementUnaryAttrs make_relu_attrs();

tl::expected<TensorShape, std::string>
    get_output_shape(ElementUnaryAttrs const &, TensorShape const &);
tl::expected<ParallelTensorShape, std::string>
    get_output_shape(ElementUnaryAttrs const &, ParallelTensorShape const &);

CHECK_VALID_OP_ATTR(ElementUnaryAttrs);

} // namespace FlexFlow

#endif
