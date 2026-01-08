#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_PERMUTE_OPEN_KWARG_DATAFLOW_GRAPH_INPUT_IDS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_PERMUTE_OPEN_KWARG_DATAFLOW_GRAPH_INPUT_IDS_H

#include "utils/containers/transform.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_open_kwarg_dataflow_graph_data.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/view_from_open_kwarg_dataflow_graph_data.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_graph_view.h"
#include "utils/overload.h"

namespace FlexFlow {

template <typename GraphInputName, typename SlotName>
OpenKwargDataflowGraphView<GraphInputName, SlotName>
    permute_open_kwarg_dataflow_graph_input_ids(
        OpenKwargDataflowGraphView<GraphInputName, SlotName> const &g,
        bidict<KwargDataflowGraphInput<GraphInputName>,
               KwargDataflowGraphInput<GraphInputName>> const
            &new_input_to_old_input) {
  std::unordered_set<KwargDataflowGraphInput<GraphInputName>> g_inputs =
      get_all_kwarg_dataflow_graph_inputs(g);
  ASSERT(g_inputs == new_input_to_old_input.right_values());

  auto new_input_from_old =
      [&](KwargDataflowGraphInput<GraphInputName> const &i)
      -> KwargDataflowGraphInput<GraphInputName> {
    return new_input_to_old_input.at_r(i);
  };

  auto new_edge_from_old =
      [&](OpenKwargDataflowEdge<GraphInputName, SlotName> const &e)
      -> OpenKwargDataflowEdge<GraphInputName, SlotName> {
    return e.template visit<
        OpenKwargDataflowEdge<GraphInputName, SlotName>>(overload{
        [&](KwargDataflowInputEdge<GraphInputName, SlotName> const &input_edge)
            -> OpenKwargDataflowEdge<GraphInputName, SlotName> {
          return OpenKwargDataflowEdge<GraphInputName, SlotName>{
              KwargDataflowInputEdge<GraphInputName, SlotName>{
                  /*src=*/new_input_from_old(input_edge.src),
                  /*dst=*/input_edge.dst,
              },
          };
        },
        [](KwargDataflowEdge<SlotName> const &standard_edge)
            -> OpenKwargDataflowEdge<GraphInputName, SlotName> {
          return OpenKwargDataflowEdge<GraphInputName, SlotName>{standard_edge};
        },
    });
  };

  OpenKwargDataflowGraphData<GraphInputName, SlotName> old_data =
      get_open_kwarg_dataflow_graph_data(g);

  OpenKwargDataflowGraphData<GraphInputName, SlotName> permuted_data =
      OpenKwargDataflowGraphData<GraphInputName, SlotName>{
          /*nodes=*/old_data.nodes,
          /*edges=*/transform(old_data.edges, new_edge_from_old),
          /*inputs=*/transform(old_data.inputs, new_input_from_old),
          /*outputs=*/old_data.outputs,
      };

  return view_from_open_kwarg_dataflow_graph_data(permuted_data);
}

} // namespace FlexFlow

#endif
