#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_CONV_2D_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_CONV_2D_H

#include "op-attrs/incoming_tensor_role.dtg.h"
#include "op-attrs/initializer_attrs.dtg.h"
#include "op-attrs/ops/conv_2d_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/singular_or_variadic.dtg.h"
#include "op-attrs/tensor_shape.h"
#include "op-attrs/tensor_slot_name.dtg.h"

namespace FlexFlow {

std::unordered_map<TensorSlotName, IncomingTensorRole>
    get_conv2d_incoming_tensor_roles(Conv2DAttrs const &);

TensorShape get_kernel_shape(Conv2DAttrs const &attrs,
                             TensorShape const &input);
TensorShape get_bias_shape(Conv2DAttrs const &attrs, TensorShape const &input);
TensorShape get_output_shape(Conv2DAttrs const &attrs,
                             TensorShape const &input);

std::unordered_map<TensorSlotName, SingularOrVariadic<TensorShape>> 
  get_weight_shapes(Conv2DAttrs const &attrs,
                    TensorShape const &input_shape);

ParallelTensorShape get_kernel_shape(Conv2DAttrs const &attrs,
                                     ParallelTensorShape const &input_shape);
ParallelTensorShape get_bias_shape(Conv2DAttrs const &attrs,
                                   ParallelTensorShape const &input_shape);
ParallelTensorShape get_output_shape(Conv2DAttrs const &attrs,
                                     ParallelTensorShape const &input_shape);

std::unordered_map<TensorSlotName, SingularOrVariadic<ParallelTensorShape>>
    get_weight_shapes(Conv2DAttrs const &attrs,
                      ParallelTensorShape const &input_shape);

std::unordered_map<TensorSlotName, InitializerAttrs> get_initializers(
    Conv2DAttrs const &attrs,
    TensorShape const &input_shape,
    std::optional<InitializerAttrs> kernel_initializer = std::nullopt,
    std::optional<InitializerAttrs> bias_initializer = std::nullopt);

} // namespace FlexFlow

#endif
