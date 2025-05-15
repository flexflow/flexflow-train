
#ifndef _FLEXFLOW_LOCAL_EXECUTION_TASK_REGISTRY_H
#define _FLEXFLOW_LOCAL_EXECUTION_TASK_REGISTRY_H

#include "local-execution/task_registry.dtg.h"
#include "op-attrs/computation_graph_op_attrs.h"
#include "pcg/computation_graph.dtg.h"
#include "task-spec/op_task_type.dtg.h"

namespace FlexFlow {

TaskRegistry construct_task_registry(
    std::unordered_map<layer_guid_t, LayerAttrs> const &);

bool registry_contains_task_for_layer(TaskRegistry const &,
                                      layer_guid_t const &,
                                      OpTaskType const &);

} // namespace FlexFlow

#endif
