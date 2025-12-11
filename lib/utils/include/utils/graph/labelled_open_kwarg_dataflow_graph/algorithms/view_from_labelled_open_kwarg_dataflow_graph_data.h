#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_VIEW_FROM_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_DATA_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_VIEW_FROM_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_DATA_H

#include "utils/graph/labelled_open_kwarg_dataflow_graph/algorithms/labelled_open_kwarg_dataflow_graph_data.dtg.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/labelled_open_kwarg_dataflow_graph_view.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/open_kwarg_dataflow_graph_data.dtg.h"
#include "utils/containers/filtrans.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/view_from_open_kwarg_dataflow_graph_data.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/algorithms/open_kwarg_dataflow_graph_view_with_labelling.h"

namespace FlexFlow {

template <typename NodeLabel,
          typename ValueLabel,
          typename GraphInputName,
          typename SlotName>
LabelledOpenKwargDataflowGraphView<NodeLabel, ValueLabel, GraphInputName, SlotName>
  view_from_labelled_open_kwarg_dataflow_graph_data(
    LabelledOpenKwargDataflowGraphData<NodeLabel, ValueLabel, GraphInputName, SlotName> const &data)
{
  OpenKwargDataflowGraphData<GraphInputName, SlotName> unlabelled_data 
    = OpenKwargDataflowGraphData<GraphInputName, SlotName>{
      /*nodes=*/keys(data.node_data),
      /*edges=*/data.edges,
      /*inputs=*/data.inputs,
      /*outputs=*/filtrans(keys(data.value_data),
                           [](OpenKwargDataflowValue<GraphInputName, SlotName> const &v) {
                             return v.try_require_internal();
                           }),
    };

  return open_kwarg_dataflow_graph_view_with_labelling(
    view_from_open_kwarg_dataflow_graph_data(unlabelled_data),
    data.node_data, 
    data.value_data);
}

} // namespace FlexFlow

#endif
