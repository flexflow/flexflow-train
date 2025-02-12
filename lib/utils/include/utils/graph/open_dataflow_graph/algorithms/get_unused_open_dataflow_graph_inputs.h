#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_ALGORITHMS_GET_UNUSED_OPEN_DATAFLOW_GRAPH_INPUTS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_ALGORITHMS_GET_UNUSED_OPEN_DATAFLOW_GRAPH_INPUTS_H

#include "utils/graph/open_dataflow_graph/open_dataflow_graph_view.h"

namespace FlexFlow {

std::unordered_set<DataflowGraphInput>
    get_unused_open_dataflow_graph_inputs(OpenDataflowGraphView const &);

} // namespace FlexFlow

#endif
