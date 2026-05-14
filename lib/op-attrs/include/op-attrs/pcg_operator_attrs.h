#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_PCG_OPERATOR_ATTRS_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_PCG_OPERATOR_ATTRS_H

#include "op-attrs/computation_graph_op_attrs.dtg.h"
#include "op-attrs/computation_graph_op_attrs.h"
#include "op-attrs/pcg_operator_attrs.dtg.h"
#include <nlohmann/json.hpp>

namespace FlexFlow {

bool is_parallel_op(PCGOperatorAttrs const &);
OperatorType pcg_op_attrs_get_op_type(PCGOperatorAttrs const &);
PCGOperatorAttrs
    pcg_op_attrs_from_compgraph_op_attrs(ComputationGraphOpAttrs const &);

nlohmann::json pcg_op_attrs_as_dot_json(PCGOperatorAttrs const &);

} // namespace FlexFlow

#endif
