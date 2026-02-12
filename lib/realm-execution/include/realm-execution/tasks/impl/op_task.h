#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_IMPL_OP_TASK_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_IMPL_OP_TASK_H

#include "kernels/profiling_settings.dtg.h"
#include "op-attrs/ops/loss_functions/loss_attrs.dtg.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "realm-execution/realm.h"
#include "realm-execution/realm_context.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.dtg.h"
#include "task-spec/ff_iteration_config.dtg.h"

namespace FlexFlow {

void op_task_body(void const *, size_t, void const *, size_t, Realm::Processor);

Realm::Event
    spawn_op_task(RealmContext &ctx,
                  Realm::Processor target_proc,
                  DynamicNodeInvocation const &invocation,
                  ProfilingSettings const &profiling_settings,
                  FFIterationConfig const &iteration_config,
                  std::optional<OptimizerAttrs> const &optimizer_attrs);

} // namespace FlexFlow

#endif
