#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_TASK_ID_T_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_TASK_ID_T_H

#include "op-attrs/pcg_operator_attrs.dtg.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "realm-execution/tasks/task_id_t.dtg.h"
#include "task-spec/dynamic_graph/dynamic_node_attrs.dtg.h"
#include <optional>
#include "realm-execution/realm.h"

namespace FlexFlow {

/**
 * \brief Retrieves the \ref task_id_t for a DynamicNodeAttrs, with
 * a return value of \ref std::nullopt to be treated as a no-op task.
 */
std::optional<task_id_t>
    get_task_id_for_op(DynamicNodeAttrs const &,
                       std::optional<OptimizerAttrs> const &);

std::optional<task_id_t>
    get_init_task_id_for_op_attrs(PCGOperatorAttrs const &);

std::optional<task_id_t> get_fwd_task_id_for_op_attrs(PCGOperatorAttrs const &);

std::optional<task_id_t> get_bwd_task_id_for_op_attrs(PCGOperatorAttrs const &);

std::optional<task_id_t>
    get_update_task_id_for_optimizer_attrs(OptimizerAttrs const &);

/**
 * \brief Convert a FlexFlow::task_id_t into a %Realm task ID.
 *
 * \relates task_id_t
 */
Realm::Processor::TaskFuncID get_realm_task_id_for_task_id(task_id_t);


} // namespace FlexFlow

#endif
