#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_KWARG_DATAFLOW_GRAPHS_ARE_ISOMORPHIC_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_KWARG_DATAFLOW_GRAPHS_ARE_ISOMORPHIC_H

#include "utils/graph/kwarg_dataflow_graph/algorithms/find_isomorphism_between_kwarg_dataflow_graphs.h"

namespace FlexFlow {

template <typename SlotName>
bool kwarg_dataflow_graphs_are_isomorphic(
    KwargDataflowGraphView<SlotName> const &lhs,
    KwargDataflowGraphView<SlotName> const &rhs) {
  std::optional<bidict<Node, Node>> found =
      find_isomorphism_between_kwarg_dataflow_graphs(lhs, rhs);

  return found.has_value();
}

} // namespace FlexFlow

#endif
