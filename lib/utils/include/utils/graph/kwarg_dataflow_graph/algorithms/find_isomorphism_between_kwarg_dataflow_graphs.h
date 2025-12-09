#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_FIND_ISOMORPHISM_BETWEEN_KWARG_DATAFLOW_GRAPHS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_FIND_ISOMORPHISM_BETWEEN_KWARG_DATAFLOW_GRAPHS_H

#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_graph_view.h"

namespace FlexFlow {

template <typename SlotName>
std::optional<bidict<Node, Node>>
  find_isomorphism_between_kwarg_dataflow_graphs(
    KwargDataflowGraphView<SlotName> const &lhs,
    KwargDataflowGraphView<SlotName> const &rhs) {
  
  std::unordered_set<OpenKwargDataflowGraphIsomorphism> open_isomorphisms = 
      find_isomorphisms_between_open_kwarg_dataflow_graphs(
        view_as_open_kwarg_dataflow_graph(lhs),
        view_as_open_kwarg_dataflow_graph(rhs));
}

} // namespace FlexFlow

#endif
