#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_OUTGOING_SLOTS_FOR_NODE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_OUTGOING_SLOTS_FOR_NODE_H

#include "utils/containers/keys.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_outgoing_kwarg_dataflow_outputs_for_node.h"

namespace FlexFlow {

template <typename SlotName>
std::unordered_set<SlotName>
    get_outgoing_slots_for_node(KwargDataflowGraphView<SlotName> const &g,
                                Node n) {
  return keys(get_outgoing_kwarg_dataflow_outputs_for_node(g, n));
}

} // namespace FlexFlow

#endif
