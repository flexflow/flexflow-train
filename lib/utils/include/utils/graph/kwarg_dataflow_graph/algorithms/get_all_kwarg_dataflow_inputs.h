#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_ALL_KWARG_DATAFLOW_INPUTS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_ALL_KWARG_DATAFLOW_INPUTS_H

#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_edge_query.h"
#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_graph_view.h"

namespace FlexFlow {

template <typename SlotName>
std::unordered_set<KwargDataflowInput<SlotName>>
    get_all_kwarg_dataflow_inputs(KwargDataflowGraphView<SlotName> const &v) {
  return transform(
      v.query_edges(kwarg_dataflow_edge_query_all<SlotName>()),
      [](KwargDataflowEdge<SlotName> const &e) -> KwargDataflowInput<SlotName> {
        return e.dst;
      });
}

} // namespace FlexFlow

#endif
