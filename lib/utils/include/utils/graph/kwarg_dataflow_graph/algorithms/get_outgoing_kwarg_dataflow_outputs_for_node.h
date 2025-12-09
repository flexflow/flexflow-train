#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_OUTGOING_KWARG_DATAFLOW_OUTPUTS_FOR_NODE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_OUTGOING_KWARG_DATAFLOW_OUTPUTS_FOR_NODE_H

#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_graph_view.h"

namespace FlexFlow {

template <typename SlotName>
std::unordered_map<SlotName, KwargDataflowOutput<SlotName>> get_outgoing_kwarg_dataflow_outputs_for_node(
  KwargDataflowGraphView<SlotName> const &g,
  Node const &n)
{
  KwargDataflowOutputQuery<SlotName> query = 
    KwargDataflowOutputQuery<SlotName>{
      /*nodes=*/query_set{n},
      /*output_idxs=*/query_set<SlotName>::matchall(),
    };

  std::unordered_map<SlotName, KwargDataflowOutput<SlotName>> result;

  for (KwargDataflowOutput<SlotName> const &output : g.query_outputs(query)) {
    result.insert({output.slot_name, output});
  }

  return result;
}

} // namespace FlexFlow

#endif
