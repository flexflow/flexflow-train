#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_ALL_KWARG_DATAFLOW_OUTPUTS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_ALL_KWARG_DATAFLOW_OUTPUTS_H

#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_graph_view.h"
#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_output_query.h"

namespace FlexFlow {

template <typename SlotName>
std::unordered_set<KwargDataflowOutput<SlotName>>
    get_all_kwarg_dataflow_outputs(
        KwargDataflowGraphView<SlotName> const &view) {
  return view.query_outputs(kwarg_dataflow_output_query_all<SlotName>());
}

} // namespace FlexFlow

#endif
