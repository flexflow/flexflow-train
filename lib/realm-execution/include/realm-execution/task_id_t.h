#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASK_ID_T_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASK_ID_T_H

#include "op-attrs/pcg_operator_attrs.dtg.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "realm-execution/task_id_t.dtg.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.dtg.h"
#include <optional>

namespace FlexFlow {

std::optional<task_id_t>
    get_task_id_for_op(DynamicNodeInvocation const &,
                       std::optional<OptimizerAttrs> const &);

std::optional<task_id_t>
    get_init_task_id_for_op_attrs(PCGOperatorAttrs const &);

std::optional<task_id_t> get_fwd_task_id_for_op_attrs(PCGOperatorAttrs const &);

std::optional<task_id_t> get_bwd_task_id_for_op_attrs(PCGOperatorAttrs const &);

std::optional<task_id_t>
    get_update_task_id_for_optimizer_attrs(OptimizerAttrs const &);

} // namespace FlexFlow

#endif
