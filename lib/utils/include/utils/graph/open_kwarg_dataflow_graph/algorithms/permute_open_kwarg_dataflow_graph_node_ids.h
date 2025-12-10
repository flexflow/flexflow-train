#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_PERMUTE_OPEN_KWARG_DATAFLOW_GRAPH_NODE_IDS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_PERMUTE_OPEN_KWARG_DATAFLOW_GRAPH_NODE_IDS_H

#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_graph_view.h"
#include "utils/overload.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_open_kwarg_dataflow_graph_data.h"

namespace FlexFlow {

template <typename GraphInputName, typename SlotName>
OpenKwargDataflowGraphView<GraphInputName, SlotName>
  permute_open_kwarg_dataflow_graph_node_ids(
    OpenKwargDataflowGraphView<GraphInputName, SlotName> const &g,
    bidict<Node, Node> const &new_node_to_old_node)
{
  auto new_node_from_old = [&](Node const &n) -> Node {
    return new_node_to_old_node.at_r(n);
  };

  auto new_output_from_old = [&](KwargDataflowOutput<SlotName> const &o) 
    -> KwargDataflowOutput<SlotName>
  {
    return KwargDataflowOutput<SlotName>{
      /*node=*/new_node_from_old(o.node),
      /*slot_name=*/o.slot_name,
    };
  };

  auto new_input_from_old = [&](KwargDataflowInput<SlotName> const &i)
    -> KwargDataflowInput<SlotName>
  {
    return KwargDataflowInput<SlotName>{
      /*node=*/new_node_from_old(i.node),
      /*slot_name=*/i.slot_name,
    };
  };

  auto new_edge_from_old = [&](OpenKwargDataflowEdge<GraphInputName, SlotName> const &e) {
    return e.template visit<
      OpenKwargDataflowEdge<GraphInputName, SlotName>
    >(overload {
      [&](KwargDataflowInputEdge<GraphInputName, SlotName> const &input_edge) {
        return OpenKwargDataflowEdge<GraphInputName, SlotName>{
          KwargDataflowInputEdge<GraphInputName, SlotName>{
            /*src=*/input_edge.src,
            /*dst=*/new_input_from_old(input_edge.dst),
          },
        };
      },
      [&](KwargDataflowEdge<SlotName> const &standard_edge) {
        return OpenKwargDataflowEdge<GraphInputName, SlotName>{
          KwargDataflowEdge<SlotName>{
            /*src=*/new_output_from_old(standard_edge.src),
            /*dst=*/new_input_from_old(standard_edge.dst),
          },
        };
      }
    });
  };

  OpenKwargDataflowGraphData<GraphInputName, SlotName> old_data = get_open_kwarg_dataflow_graph_data(g);

  OpenKwargDataflowGraphData<GraphInputName, SlotName> permuted_data = 
    OpenKwargDataflowGraphData<GraphInputName, SlotName>{
    /*nodes=*/transform(old_data.nodes, new_node_from_old),
    /*edges=*/transform(old_data.edges, new_edge_from_old),
    /*inputs=*/old_data.inputs,
    /*outputs=*/transform(old_data.outputs, new_output_from_old),
  };

  return view_from_open_dataflow_graph_data(permuted_data);
}
 
} // namespace FlexFlow

#endif
