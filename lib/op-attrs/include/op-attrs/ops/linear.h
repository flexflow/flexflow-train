#ifndef _FLEXFLOW_LINEAR_ATTRS_H
#define _FLEXFLOW_LINEAR_ATTRS_H

#include "op-attrs/incoming_tensor_role.dtg.h"
#include "op-attrs/ops/core.h"
#include "op-attrs/ops/linear_attrs.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_num_dims.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "utils/record_formatter.h"
#include <tl/expected.hpp>
#include "op-attrs/operator_space_parallel_tensor_space_mapping.dtg.h"
#include "op-attrs/parallel_tensor_space_mapping.dtg.h"

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

tl::expected<ParallelTensorSpaceMapping, std::string>
    get_projection_to_output_parallel_dim_mapping(LinearAttrs const &attrs, 
                                                  ParallelTensorDimDegrees const &input);

tl::expected<ParallelTensorDimDegrees, std::string>
    get_projection_parallel_dim_degrees(LinearAttrs const &attrs, ParallelTensorDimDegrees const &input);
tl::expected<ParallelTensorDimDegrees, std::string>
    get_bias_parallel_dim_degrees(LinearAttrs const &attrs, ParallelTensorDimDegrees const &input);
tl::expected<ParallelTensorDimDegrees, std::string>
    get_output_parallel_dim_degrees(LinearAttrs const &attrs, ParallelTensorDimDegrees const &input);

tl::expected<ParallelTensorShape, std::string>
    get_projection_shape(LinearAttrs const &attrs,
                         ParallelTensorShape const &input);
tl::expected<ParallelTensorShape, std::string>
    get_bias_shape(LinearAttrs const &attrs, ParallelTensorShape const &input);
tl::expected<ParallelTensorShape, std::string>
    get_output_shape(LinearAttrs const &attrs,
                     ParallelTensorShape const &input);

tl::expected<OperatorSpaceParallelTensorSpaceMapping, std::string>
    get_projection_space_mapping(LinearAttrs const &attrs,
                                 ParallelTensorDimDegrees const &input);
tl::expected<OperatorSpaceParallelTensorSpaceMapping, std::string>
    get_bias_space_mapping(LinearAttrs const &attrs,
                           ParallelTensorDimDegrees const &input);
tl::expected<OperatorSpaceParallelTensorSpaceMapping, std::string>
    get_output_space_mapping(LinearAttrs const &attrs, 
                             TensorNumDims const &input_num_dims);

} // namespace FlexFlow

#endif
