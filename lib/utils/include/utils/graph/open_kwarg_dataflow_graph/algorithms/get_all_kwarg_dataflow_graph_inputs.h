#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_ALL_KWARG_DATAFLOW_GRAPH_INPUTS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_ALL_KWARG_DATAFLOW_GRAPH_INPUTS_H

#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_graph_view.h"

namespace FlexFlow {

template <typename GraphInputName, typename SlotName>
std::unordered_set<KwargDataflowGraphInput<GraphInputName>>
    get_all_kwarg_dataflow_graph_inputs(
        OpenKwargDataflowGraphView<GraphInputName, SlotName> const &view) {
  return view.get_inputs();
}

} // namespace FlexFlow

#endif
