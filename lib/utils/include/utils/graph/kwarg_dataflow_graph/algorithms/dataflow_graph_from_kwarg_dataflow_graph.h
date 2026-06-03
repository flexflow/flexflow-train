#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_DATAFLOW_GRAPH_FROM_KWARG_DATAFLOW_GRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_DATAFLOW_GRAPH_FROM_KWARG_DATAFLOW_GRAPH_H

#include "utils/graph/dataflow_graph/algorithms/view_from_dataflow_graph_data.h"
#include "utils/graph/dataflow_graph/dataflow_graph_view.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/dataflow_graph_data_from_kwarg_dataflow_graph_data.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_kwarg_dataflow_graph_data.h"
#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_graph_view.h"

namespace FlexFlow {

template <typename SlotName>
DataflowGraphView dataflow_graph_from_kwarg_dataflow_graph(
    KwargDataflowGraphView<SlotName> const &kwarg_dg,
    std::function<std::vector<SlotName>(
        std::unordered_set<SlotName> const &)> const &order_slots) {
  KwargDataflowGraphData<SlotName> kwarg_data =
      get_kwarg_dataflow_graph_data(kwarg_dg);

  DataflowGraphData result_data =
      dataflow_graph_data_from_kwarg_dataflow_graph_data(kwarg_data,
                                                         order_slots);

  return view_from_dataflow_graph_data(result_data);
}

} // namespace FlexFlow

#endif
