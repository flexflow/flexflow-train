#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_OPEN_KWARG_DATAFLOW_GRAPHS_ARE_ISOMORPHIC_UNDER_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_OPEN_KWARG_DATAFLOW_GRAPHS_ARE_ISOMORPHIC_UNDER_H

#include "utils/graph/open_kwarg_dataflow_graph/algorithms/open_kwarg_dataflow_graph_isomorphism.dtg.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_graph_view.h"

namespace FlexFlow {

template <typename GraphInputName, typename SlotName>
bool open_kwarg_dataflow_graphs_are_isomorphic_under(
  OpenKwargDataflowGraphView<GraphInputName, SlotName> const &src,
  OpenKwargDataflowGraphView<GraphInputName, SlotName> const &dst,
  OpenKwargDataflowGraphIsomorphism<GraphInputName> const &isomorphism)
{
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow

#endif
