#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_DYNAMIC_GRAPH_EDGE_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_DYNAMIC_GRAPH_EDGE_H

#include "task-spec/dynamic_graph/dynamic_graph_edge.dtg.h"
#include "task-spec/dynamic_graph/dynamic_slot_site.dtg.h"

namespace FlexFlow {

DynamicGraphEdge
    dynamic_graph_edge_from_slot_sites(DynamicSlotSite const &src,
                                       InternalDynamicSlotSite const &dst);

} // namespace FlexFlow

#endif
