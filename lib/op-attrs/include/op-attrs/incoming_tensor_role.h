#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_INCOMING_TENSOR_ROLE_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_INCOMING_TENSOR_ROLE_H

#include "op-attrs/incoming_tensor_role.dtg.h"
#include "op-attrs/tensor_role.dtg.h"

namespace FlexFlow {

TensorRole tensor_role_from_incoming_tensor_role(IncomingTensorRole);

} // namespace FlexFlow

#endif
