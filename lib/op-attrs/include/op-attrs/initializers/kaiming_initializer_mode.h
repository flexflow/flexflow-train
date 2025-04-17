#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_INITIALIZERS_KAIMING_INITIALIZER_MODE_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_INITIALIZERS_KAIMING_INITIALIZER_MODE_H

#include "op-attrs/initializers/kaiming_initializer_mode.dtg.h"
#include "op-attrs/tensor_dims.dtg.h"
#include "utils/nonnegative_int/nonnegative_int.h"

namespace FlexFlow {

/**
 * @brief `fan_in` and `fan_out` calculation from pytorch
 *
 * see
 * https://github.com/pytorch/pytorch/blob/bd019c0bb485904a99fb38589444b1461ab1e486/torch/nn/init.py#L345-L363
 */
nonnegative_int calculate_fan_for_mode(TensorDims const &dims,
                                       KaimingInitializerMode mode);

} // namespace FlexFlow

#endif
