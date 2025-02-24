#ifndef _FLEXFLOW_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_BATCH_NORM_H
#define _FLEXFLOW_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_BATCH_NORM_H

#include "op-attrs/incoming_tensor_role.dtg.h"
#include "op-attrs/initializer_attrs.dtg.h"
#include "op-attrs/ops/batch_norm_attrs.dtg.h"
#include "op-attrs/ops/core.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"

namespace FlexFlow {

std::vector<IncomingTensorRole>
    get_batch_norm_incoming_tensor_roles(BatchNormAttrs const &);

tl::expected<TensorShape, std::string> get_output_shape(BatchNormAttrs const &,
                                                        TensorShape const &);
tl::expected<TensorShape, std::string>
    get_gamma_weights_shape(BatchNormAttrs const &, TensorShape const &);
tl::expected<TensorShape, std::string>
    get_beta_weights_shape(BatchNormAttrs const &, TensorShape const &);

tl::expected<std::vector<TensorShape>, std::string>
    get_weight_shapes(BatchNormAttrs const &attrs,
                      TensorShape const &input_shape);

tl::expected<ParallelTensorDimDegrees, std::string>
    get_output_parallel_dim_degrees(BatchNormAttrs const &,
                                    ParallelTensorDimDegrees const &);
tl::expected<ParallelTensorDimDegrees, std::string>
    get_gamma_weights_parallel_dim_degrees(BatchNormAttrs const &,
                                           ParallelTensorDimDegrees const &);
tl::expected<ParallelTensorDimDegrees, std::string>
    get_beta_weights_parallel_dim_degrees(BatchNormAttrs const &,
                                          ParallelTensorDimDegrees const &);

tl::expected<std::vector<ParallelTensorDimDegrees>, std::string>
    get_weight_parallel_dim_degrees(
        BatchNormAttrs const &attrs,
        ParallelTensorDimDegrees const &input_degrees);

tl::expected<ParallelTensorShape, std::string>
    get_output_shape(BatchNormAttrs const &, ParallelTensorShape const &);
tl::expected<ParallelTensorShape, std::string>
    get_gamma_weights_shape(BatchNormAttrs const &,
                            ParallelTensorShape const &);
tl::expected<ParallelTensorShape, std::string>
    get_beta_weights_shape(BatchNormAttrs const &, ParallelTensorShape const &);

tl::expected<std::vector<ParallelTensorShape>, std::string>
    get_weight_shapes(BatchNormAttrs const &attrs,
                      ParallelTensorShape const &input_shape);

/**
 * @brief Chosen to match pytorch
 *
 * see
 * https://github.com/pytorch/pytorch/blob/1eba9b3aa3c43f86f4a2c807ac8e12c4a7767340/torch/nn/modules/batchnorm.py#L93-L97
 */
tl::expected<std::vector<InitializerAttrs>, std::string>
    get_initializers(BatchNormAttrs const &attrs);

CHECK_VALID_OP_ATTR(BatchNormAttrs);

} // namespace FlexFlow

#endif
