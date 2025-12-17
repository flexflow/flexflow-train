#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_DATA_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_DATA_H

#include "utils/graph/labelled_open_kwarg_dataflow_graph/algorithms/labelled_open_kwarg_dataflow_graph_data.dtg.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/open_kwarg_dataflow_graph_data.dtg.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/open_kwarg_dataflow_graph_data.h"
#include "utils/containers/keys.h"
#include "utils/containers/filtrans.h"

namespace FlexFlow {

template <typename NodeLabel,
          typename ValueLabel,
          typename GraphInputName,
          typename SlotName>
OpenKwargDataflowGraphData<GraphInputName, SlotName>
  labelled_open_kwarg_dataflow_graph_data_without_labels(
    LabelledOpenKwargDataflowGraphData<NodeLabel, ValueLabel, GraphInputName, SlotName> const &labelled_data)
{
  OpenKwargDataflowGraphData<GraphInputName, SlotName> result =
    OpenKwargDataflowGraphData<GraphInputName, SlotName>{
      /*nodes=*/keys(labelled_data.node_data),
      /*edges=*/labelled_data.edges,
      /*inputs=*/labelled_data.inputs,
      /*outputs=*/filtrans(keys(labelled_data.value_data),
                           [](OpenKwargDataflowValue<GraphInputName, SlotName> const &v) {
                             return v.try_require_internal();
                           }),
    };

  require_open_kwarg_dataflow_graph_data_is_valid(result);

  return result;
}

template <typename NodeLabel,
          typename ValueLabel,
          typename GraphInputName,
          typename SlotName>
void require_labelled_open_kwarg_dataflow_graph_data_is_valid(
  LabelledOpenKwargDataflowGraphData<NodeLabel, ValueLabel, GraphInputName, SlotName> const &data)
{
  labelled_open_kwarg_dataflow_graph_data_without_labels(data);  
}

} // namespace FlexFlow

#endif
