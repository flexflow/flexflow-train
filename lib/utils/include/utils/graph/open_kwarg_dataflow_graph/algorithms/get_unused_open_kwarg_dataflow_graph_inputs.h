#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_UNUSED_OPEN_KWARG_DATAFLOW_GRAPH_INPUTS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_UNUSED_OPEN_KWARG_DATAFLOW_GRAPH_INPUTS_H

#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_all_kwarg_dataflow_graph_inputs.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_open_kwarg_dataflow_value_uses.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_graph_view.h"

namespace FlexFlow {

template <typename GraphInputName, typename SlotName>
std::unordered_set<KwargDataflowGraphInput<GraphInputName>>
    get_unused_open_kwarg_dataflow_graph_inputs(
        OpenKwargDataflowGraphView<GraphInputName, SlotName> const &g) {
  return filter(
      get_all_kwarg_dataflow_graph_inputs(g),
      [&](KwargDataflowGraphInput<GraphInputName> const &i) {
        return get_open_kwarg_dataflow_value_uses(
                   g, OpenKwargDataflowValue<GraphInputName, SlotName>{i})
            .empty();
      });
}

} // namespace FlexFlow

#endif
