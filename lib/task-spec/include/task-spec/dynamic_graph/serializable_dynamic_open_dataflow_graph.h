#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_SERIALIZABLE_DYNAMIC_OPEN_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_SERIALIZABLE_DYNAMIC_OPEN_DATAFLOW_GRAPH_H

#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/serializable_dynamic_open_dataflow_graph.dtg.h"

namespace FlexFlow {

SerializableDynamicOpenDataflowGraph
    dynamic_open_dataflow_graph_to_serializable(
        DynamicOpenDataflowGraph const &);
DynamicOpenDataflowGraph dynamic_open_dataflow_graph_from_serializable(
    SerializableDynamicOpenDataflowGraph const &);

} // namespace FlexFlow

#endif
