#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_LABELLED_KWARG_DATAFLOW_GRAPH_DATA_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_LABELLED_KWARG_DATAFLOW_GRAPH_DATA_H

#include "utils/graph/kwarg_dataflow_graph/algorithms/get_all_kwarg_dataflow_edges.h"
#include "utils/graph/labelled_kwarg_dataflow_graph/algorithms/get_labelled_kwarg_dataflow_graph_node_label_map.h"
#include "utils/graph/labelled_kwarg_dataflow_graph/algorithms/get_labelled_kwarg_dataflow_graph_output_label_map.h"
#include "utils/graph/labelled_kwarg_dataflow_graph/algorithms/labelled_kwarg_dataflow_graph_data.dtg.h"
#include "utils/graph/labelled_kwarg_dataflow_graph/labelled_kwarg_dataflow_graph_view.h"

namespace FlexFlow {

template <typename NodeLabel, typename ValueLabel, typename SlotName>
LabelledKwargDataflowGraphData<NodeLabel, ValueLabel, SlotName>
    get_labelled_kwarg_dataflow_graph_data(
        LabelledKwargDataflowGraphView<NodeLabel, ValueLabel, SlotName> const
            &g) {
  return LabelledKwargDataflowGraphData<NodeLabel, ValueLabel, SlotName>{
      /*node_data=*/get_labelled_kwarg_dataflow_graph_node_label_map(g),
      /*edges=*/get_all_kwarg_dataflow_edges(g),
      /*output_data=*/get_labelled_kwarg_dataflow_graph_output_label_map(g),
  };
}

} // namespace FlexFlow

#endif
