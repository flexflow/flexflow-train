#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_OPEN_KWARG_DATAFLOW_GRAPH_DATA_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_OPEN_KWARG_DATAFLOW_GRAPH_DATA_H

#include "utils/graph/kwarg_dataflow_graph/algorithms/get_all_kwarg_dataflow_outputs.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_all_kwarg_dataflow_graph_inputs.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_all_open_kwarg_dataflow_edges.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/open_kwarg_dataflow_graph_data.dtg.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_graph_view.h"

namespace FlexFlow {

template <typename GraphInputName, typename SlotName>
OpenKwargDataflowGraphData<GraphInputName, SlotName>
    get_open_kwarg_dataflow_graph_data(
        OpenKwargDataflowGraphView<GraphInputName, SlotName> const &g) {
  return OpenKwargDataflowGraphData<GraphInputName, SlotName>{
      /*nodes=*/get_nodes(g),
      /*edges=*/get_all_open_kwarg_dataflow_edges(g),
      /*inputs=*/get_all_kwarg_dataflow_graph_inputs(g),
      /*outputs=*/get_all_kwarg_dataflow_outputs(g),
  };
}

} // namespace FlexFlow

#endif
