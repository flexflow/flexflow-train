#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_DYNAMIC_NODE_MAPPING_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_DYNAMIC_NODE_MAPPING_H

#include "task-spec/dynamic_graph/dynamic_node_mapping.dtg.h"
#include "task-spec/global_device_id_t.dtg.h"

namespace FlexFlow {

bidict<ParallelTensorSpaceCoordinate, global_device_id_t>
    dynamic_node_mapping_bindings_for_slot_name(DynamicNodeMapping const &,
                                                TensorSlotName const &);

std::unordered_set<global_device_id_t>
    target_devices_of_dynamic_node_mapping(DynamicNodeMapping const &);

} // namespace FlexFlow

#endif
