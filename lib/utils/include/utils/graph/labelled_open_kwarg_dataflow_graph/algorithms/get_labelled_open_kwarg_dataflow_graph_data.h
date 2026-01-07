#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_DATA_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_DATA_H

#include "utils/containers/generate_map.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/algorithms/labelled_open_kwarg_dataflow_graph_data.dtg.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/labelled_open_kwarg_dataflow_graph_view.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_all_kwarg_dataflow_graph_inputs.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_all_open_kwarg_dataflow_edges.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_all_open_kwarg_dataflow_values.h"

namespace FlexFlow {

template <typename NodeLabel,
          typename ValueLabel,
          typename GraphInputName,
          typename SlotName>
LabelledOpenKwargDataflowGraphData<NodeLabel,
                                   ValueLabel,
                                   GraphInputName,
                                   SlotName>
    get_labelled_open_kwarg_dataflow_graph_data(
        LabelledOpenKwargDataflowGraphView<NodeLabel,
                                           ValueLabel,
                                           GraphInputName,
                                           SlotName> const &g) {
  return LabelledOpenKwargDataflowGraphData<NodeLabel,
                                            ValueLabel,
                                            GraphInputName,
                                            SlotName>{
      /*nodes=*/generate_map(
          get_nodes(g), [&](Node const &n) -> NodeLabel { return g.at(n); }),
      /*edges=*/get_all_open_kwarg_dataflow_edges(g),
      /*inputs=*/get_all_kwarg_dataflow_graph_inputs(g),
      /*outputs=*/
      generate_map(
          get_all_open_kwarg_dataflow_values(g),
          [&](OpenKwargDataflowValue<GraphInputName, SlotName> const &v)
              -> ValueLabel { return g.at(v); }),
  };
}

} // namespace FlexFlow

#endif
