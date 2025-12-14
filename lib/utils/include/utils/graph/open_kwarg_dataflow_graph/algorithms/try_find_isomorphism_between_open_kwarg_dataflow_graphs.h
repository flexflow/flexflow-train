#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_TRY_FIND_ISOMORPHISM_BETWEEN_OPEN_KWARG_DATAFLOW_GRAPHS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_TRY_FIND_ISOMORPHISM_BETWEEN_OPEN_KWARG_DATAFLOW_GRAPHS_H

#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_graph_view.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/find_isomorphisms_between_open_kwarg_dataflow_graphs.h"
#include "utils/containers/try_get_one_of.h"

namespace FlexFlow {

template <typename GraphInputName, typename SlotName>
std::optional<OpenKwargDataflowGraphIsomorphism<GraphInputName>>
    try_find_isomorphism_between_open_kwarg_dataflow_graphs(
        OpenKwargDataflowGraphView<GraphInputName, SlotName> const &src,
        OpenKwargDataflowGraphView<GraphInputName, SlotName> const &dst)
{
  std::unordered_set<OpenKwargDataflowGraphIsomorphism<GraphInputName>>
    isomorphisms = find_isomorphisms_between_open_kwarg_dataflow_graphs(src, dst);

  return try_get_one_of(isomorphisms);
}

} // namespace FlexFlow

#endif
