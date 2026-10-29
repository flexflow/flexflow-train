#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_GET_TENSOR_ROLES_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_GET_TENSOR_ROLES_H

#include "op-attrs/computation_graph_op_attrs.dtg.h"
#include "op-attrs/pcg_operator_attrs.dtg.h"
#include "op-attrs/tensor_role.dtg.h"
#include "op-attrs/tensor_slot_name.dtg.h"

namespace FlexFlow {

std::map<TensorSlotName, TensorRole> get_tensor_roles(PCGOperatorAttrs const &);

} // namespace FlexFlow

#endif
