#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_LABELLED_OPEN_KWARG_DATAFLOW_GRAPHS_ARE_ISOMORPHIC_UNDER_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_LABELLED_OPEN_KWARG_DATAFLOW_GRAPHS_ARE_ISOMORPHIC_UNDER_H

#include "utils/graph/labelled_open_kwarg_dataflow_graph/algorithms/labelled_open_kwarg_dataflow_graph_data.dtg.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/labelled_open_kwarg_dataflow_graph_view.h"
#include "utils/graph/node/algorithms/new_node.dtg.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/open_kwarg_dataflow_graph_isomorphism.dtg.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/algorithms/get_labelled_open_kwarg_dataflow_graph_data.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/algorithms/permute_labelled_open_kwarg_dataflow_graph_node_ids.h"
#include "utils/bidict/algorithms/transform_values.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/algorithms/permute_labelled_open_kwarg_dataflow_graph_input_ids.h"

namespace FlexFlow {

template <typename NodeLabel,
          typename ValueLabel,
          typename GraphInputName, 
          typename SlotName>
bool labelled_open_kwarg_dataflow_graphs_are_isomorphic_under(
  LabelledOpenKwargDataflowGraphView<NodeLabel, ValueLabel, GraphInputName, SlotName> const &src,
  LabelledOpenKwargDataflowGraphView<NodeLabel, ValueLabel, GraphInputName, SlotName> const &dst,
  OpenKwargDataflowGraphIsomorphism<GraphInputName> const &candidate_isomorphism)
{
  bidict<NewNode, Node> new_node_to_old_node = 
    transform_values(candidate_isomorphism.node_mapping, 
                     [](Node const &n) {
                       return NewNode{n};    
                     }).reversed();

  bidict<KwargDataflowGraphInput<GraphInputName>, KwargDataflowGraphInput<GraphInputName>> 
    new_input_to_old_input = candidate_isomorphism.input_mapping.reversed();

  LabelledOpenKwargDataflowGraphData<NodeLabel, ValueLabel, GraphInputName, SlotName> permuted_data =
    get_labelled_open_kwarg_dataflow_graph_data(
      permute_labelled_open_kwarg_dataflow_graph_input_ids(
        permute_labelled_open_kwarg_dataflow_graph_node_ids(
          src,
          new_node_to_old_node),
        new_input_to_old_input));

  LabelledOpenKwargDataflowGraphData<NodeLabel, ValueLabel, GraphInputName, SlotName> dst_data = 
    get_labelled_open_kwarg_dataflow_graph_data(dst);

  return permuted_data == dst_data;
}

} // namespace FlexFlow

#endif
