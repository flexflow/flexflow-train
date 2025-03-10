#ifndef _FLEXFLOW_LINEAR_ATTRS_H
#define _FLEXFLOW_LINEAR_ATTRS_H

#include "op-attrs/incoming_tensor_role.dtg.h"
#include "op-attrs/initializer_attrs.dtg.h"
#include "op-attrs/ops/core.h"
#include "op-attrs/ops/linear_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "utils/record_formatter.h"
#include <tl/expected.hpp>

namespace FlexFlow {

std::vector<IncomingTensorRole>
    get_linear_incoming_tensor_roles(LinearAttrs const &);

CHECK_VALID_OP_ATTR(LinearAttrs);

RecordFormatter as_dot(LinearAttrs const &);

tl::expected<TensorShape, std::string>
    get_projection_shape(LinearAttrs const &attrs, TensorShape const &input);
tl::expected<TensorShape, std::string> get_bias_shape(LinearAttrs const &attrs,
                                                      TensorShape const &input);
tl::expected<TensorShape, std::string>
    get_output_shape(LinearAttrs const &attrs, TensorShape const &input);

tl::expected<std::vector<TensorShape>, std::string>
    get_weight_shapes(LinearAttrs const &attrs, TensorShape const &input_shape);

tl::expected<ParallelTensorShape, std::string>
    get_projection_shape(LinearAttrs const &attrs,
                         ParallelTensorShape const &input);
tl::expected<ParallelTensorShape, std::string>
    get_bias_shape(LinearAttrs const &attrs, ParallelTensorShape const &input);
tl::expected<ParallelTensorShape, std::string>
    get_output_shape(LinearAttrs const &attrs,
                     ParallelTensorShape const &input);

tl::expected<std::vector<ParallelTensorShape>, std::string>
    get_weight_shapes(LinearAttrs const &attrs,
                      ParallelTensorShape const &input_shape);

tl::expected<std::vector<InitializerAttrs>, std::string> get_initializers(
    LinearAttrs const &,
    TensorShape const &input_shape,
    std::optional<InitializerAttrs> const &projection_initializer =
        std::nullopt,
    std::optional<InitializerAttrs> const &kernel_initializer = std::nullopt);

} // namespace FlexFlow

#endif
