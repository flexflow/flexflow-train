#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_FIND_ISOMORPHISM_BETWEEN_LABELLED_OPEN_KWARG_DATAFLOW_GRAPHS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_FIND_ISOMORPHISM_BETWEEN_LABELLED_OPEN_KWARG_DATAFLOW_GRAPHS_H

#include "utils/graph/labelled_open_kwarg_dataflow_graph/labelled_open_kwarg_dataflow_graph_view.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/open_kwarg_dataflow_graph_isomorphism.dtg.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/algorithms/labelled_open_kwarg_dataflow_graphs_are_isomorphic_under.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/find_isomorphisms_between_open_kwarg_dataflow_graphs.h"

namespace FlexFlow {

template <typename NodeLabel,
          typename ValueLabel, 
          typename GraphInputName,
          typename SlotName>
std::optional<OpenKwargDataflowGraphIsomorphism<GraphInputName>>
  find_isomorphism_between_labelled_open_kwarg_dataflow_graphs(
    LabelledOpenKwargDataflowGraphView<NodeLabel, ValueLabel, GraphInputName, SlotName> const &src,
    LabelledOpenKwargDataflowGraphView<NodeLabel, ValueLabel, GraphInputName, SlotName> const &dst)
{
  std::unordered_set<OpenKwargDataflowGraphIsomorphism<GraphInputName>>
    unlabelled_isomorphisms = find_isomorphisms_between_open_kwarg_dataflow_graphs(
      static_cast<OpenKwargDataflowGraphView<GraphInputName, SlotName>>(src),
      static_cast<OpenKwargDataflowGraphView<GraphInputName, SlotName>>(dst));

   
  for (OpenKwargDataflowGraphIsomorphism<GraphInputName> const &candidate_isomorphism :
       unlabelled_isomorphisms) {
    if (labelled_open_kwarg_dataflow_graphs_are_isomorphic_under(src, dst, candidate_isomorphism)) {
      return candidate_isomorphism;
    }
  }

  return std::nullopt;
}

} // namespace FlexFlow

#endif
