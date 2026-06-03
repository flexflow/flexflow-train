#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_OPEN_KWARG_DATAFLOW_GRAPH_DATA_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_OPEN_KWARG_DATAFLOW_GRAPH_DATA_H

#include "utils/containers/filtrans.h"
#include "utils/containers/is_subseteq_of.h"
#include "utils/containers/transform.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/kwarg_dataflow_graph_data.dtg.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/kwarg_dataflow_graph_data.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/open_kwarg_dataflow_graph_data.dtg.h"

namespace FlexFlow {

template <typename GraphInputName, typename SlotName>
void require_open_kwarg_dataflow_graph_data_is_valid(
    OpenKwargDataflowGraphData<GraphInputName, SlotName> const &data) {
  std::unordered_set<KwargDataflowGraphInput<GraphInputName>>
      inputs_from_edges = filtrans(
          data.edges,
          [](OpenKwargDataflowEdge<GraphInputName, SlotName> const &e)
              -> std::optional<KwargDataflowGraphInput<GraphInputName>> {
            return transform(
                e.try_require_input_edge(),
                [](KwargDataflowInputEdge<GraphInputName, SlotName> const
                       &input_e) -> KwargDataflowGraphInput<GraphInputName> {
                  return input_e.src;
                });
          });

  ASSERT(is_subseteq_of(inputs_from_edges, data.inputs));

  require_kwarg_dataflow_graph_data_is_valid(
      kwarg_dataflow_graph_data_from_open(data));
}

template <typename GraphInputName, typename SlotName>
KwargDataflowGraphData<SlotName> kwarg_dataflow_graph_data_from_open(
    OpenKwargDataflowGraphData<GraphInputName, SlotName> const &open_data) {

  return KwargDataflowGraphData{
      /*nodes=*/open_data.nodes,
      /*edges=*/
      filtrans(
          open_data.edges,
          [](OpenKwargDataflowEdge<GraphInputName, SlotName> const &open_edge)
              -> std::optional<KwargDataflowEdge<SlotName>> {
            return open_edge.try_require_internal_edge();
          }),
      /*outputs=*/open_data.outputs,
  };
}

} // namespace FlexFlow

#endif
