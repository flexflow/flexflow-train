#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_DYNAMIC_NODE_INVOCATION_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_DYNAMIC_NODE_INVOCATION_H

#include "pcg/tensor_direction.dtg.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.dtg.h"
#include "task-spec/dynamic_graph/dynamic_slot_site.dtg.h"
#include "task-spec/dynamic_graph/training_op_type.dtg.h"

namespace FlexFlow {

std::unordered_map<DynamicTensorSlot, DynamicValueAttrs>
    get_slot_map_for_direction(DynamicNodeInvocation const &, TensorDirection);

TrainingOpType
    dynamic_node_invocation_get_op_type(DynamicNodeInvocation const &);

std::unordered_set<InternalDynamicSlotSite>
    get_dynamic_slot_sites_for_invocation(DynamicNodeInvocation const &);

} // namespace FlexFlow

#endif
