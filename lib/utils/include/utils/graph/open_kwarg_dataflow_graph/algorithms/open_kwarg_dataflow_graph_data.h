#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_OPEN_KWARG_DATAFLOW_GRAPH_DATA_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_OPEN_KWARG_DATAFLOW_GRAPH_DATA_H

#include "utils/containers/filtrans.h"
#include "utils/containers/is_subseteq_of.h"
#include "utils/containers/transform.h"
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
}

} // namespace FlexFlow

#endif
