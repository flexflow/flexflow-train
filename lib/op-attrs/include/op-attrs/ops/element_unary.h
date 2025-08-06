#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_ELEMENT_UNARY_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_ELEMENT_UNARY_H

#include "op-attrs/num_ptensor_parallel_dims_t.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_mapping.dtg.h"
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

OperatorSpaceToParallelTensorSpaceMapping get_operator_to_input_mapping(
  ElementUnaryAttrs const &attrs,
  num_ptensor_parallel_dims_t input_num_dims);

OperatorSpaceToParallelTensorSpaceMapping get_operator_to_output_mapping(
  ElementUnaryAttrs const &attrs,
  num_ptensor_parallel_dims_t input_num_dims);


} // namespace FlexFlow

#endif
