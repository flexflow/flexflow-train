#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_FIND_ISOMORPHISM_BETWEEN_KWARG_DATAFLOW_GRAPHS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_FIND_ISOMORPHISM_BETWEEN_KWARG_DATAFLOW_GRAPHS_H

#include "utils/containers/get_one_of.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/view_as_open_kwarg_dataflow_graph.h"
#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_graph_view.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/find_isomorphisms_between_open_kwarg_dataflow_graphs.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/open_kwarg_dataflow_graph_isomorphism.dtg.h"

namespace FlexFlow {

template <typename SlotName>
std::optional<bidict<Node, Node>>
    find_isomorphism_between_kwarg_dataflow_graphs(
        KwargDataflowGraphView<SlotName> const &lhs,
        KwargDataflowGraphView<SlotName> const &rhs) {

  std::unordered_set<OpenKwargDataflowGraphIsomorphism<int>> open_isomorphisms =
      find_isomorphisms_between_open_kwarg_dataflow_graphs(
          view_as_open_kwarg_dataflow_graph<int, SlotName>(lhs),
          view_as_open_kwarg_dataflow_graph<int, SlotName>(rhs));

  if (open_isomorphisms.empty()) {
    return std::nullopt;
  } else {
    OpenKwargDataflowGraphIsomorphism<int> chosen =
        get_one_of(open_isomorphisms);
    ASSERT(chosen.input_mapping.empty());
    return chosen.node_mapping;
  }
}

} // namespace FlexFlow

#endif
