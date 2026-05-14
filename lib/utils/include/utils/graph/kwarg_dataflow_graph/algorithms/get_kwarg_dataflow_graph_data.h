#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_KWARG_DATAFLOW_GRAPH_DATA_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_KWARG_DATAFLOW_GRAPH_DATA_H

#include "utils/graph/kwarg_dataflow_graph/algorithms/get_all_kwarg_dataflow_edges.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_all_kwarg_dataflow_outputs.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/kwarg_dataflow_graph_data.dtg.h"
#include "utils/graph/node/algorithms.h"

namespace FlexFlow {

template <typename SlotName>
KwargDataflowGraphData<SlotName>
    get_kwarg_dataflow_graph_data(KwargDataflowGraphView<SlotName> const &g) {
  return KwargDataflowGraphData<SlotName>{
      /*nodes=*/get_nodes(g),
      /*edges=*/get_all_kwarg_dataflow_edges(g),
      /*outputs=*/get_all_kwarg_dataflow_outputs(g),
  };
}

} // namespace FlexFlow

#endif
