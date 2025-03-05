#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_INITIALIZER_ATTRS_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_INITIALIZER_ATTRS_H

#include "op-attrs/initializer_attrs.dtg.h"
#include "op-attrs/tensor_dims.dtg.h"

namespace FlexFlow {

InitializerAttrs make_zero_initializer();
InitializerAttrs make_kaiming_uniform(TensorDims const &);

} // namespace FlexFlow

#endif
