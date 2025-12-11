#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_OPEN_KWARG_DATAFLOW_GRAPHS_ARE_ISOMORPHIC_UNDER_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_OPEN_KWARG_DATAFLOW_GRAPHS_ARE_ISOMORPHIC_UNDER_H

#include "utils/graph/open_kwarg_dataflow_graph/algorithms/open_kwarg_dataflow_graph_data.dtg.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/open_kwarg_dataflow_graph_isomorphism.dtg.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_graph_view.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/permute_open_kwarg_dataflow_graph_node_ids.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/permute_open_kwarg_dataflow_graph_input_ids.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_open_kwarg_dataflow_graph_data.h"
#include "utils/bidict/algorithms/transform_values.h"

namespace FlexFlow {

template <typename GraphInputName, typename SlotName>
bool open_kwarg_dataflow_graphs_are_isomorphic_under(
  OpenKwargDataflowGraphView<GraphInputName, SlotName> const &src,
  OpenKwargDataflowGraphView<GraphInputName, SlotName> const &dst,
  OpenKwargDataflowGraphIsomorphism<GraphInputName> const &isomorphism)
{
  bidict<NewNode, Node> new_node_to_old_node = 
    transform_values(isomorphism.node_mapping, 
                     [](Node const &n) {
                       return NewNode{n};    
                     }).reversed();

  bidict<KwargDataflowGraphInput<GraphInputName>, KwargDataflowGraphInput<GraphInputName>> 
    new_input_to_old_input = isomorphism.input_mapping.reversed();

  OpenKwargDataflowGraphData<GraphInputName, SlotName> permuted_data =
    get_open_kwarg_dataflow_graph_data(
      permute_open_kwarg_dataflow_graph_input_ids(
        permute_open_kwarg_dataflow_graph_node_ids(
          src,
          new_node_to_old_node),
        new_input_to_old_input));

  OpenKwargDataflowGraphData<GraphInputName, SlotName> dst_data = 
    get_open_kwarg_dataflow_graph_data(dst);

  return permuted_data == dst_data;
}

} // namespace FlexFlow

#endif
