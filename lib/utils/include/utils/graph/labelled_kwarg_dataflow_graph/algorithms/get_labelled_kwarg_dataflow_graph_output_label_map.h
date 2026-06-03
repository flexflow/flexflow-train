#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_LABELLED_KWARG_DATAFLOW_GRAPH_OUTPUT_LABEL_MAP_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_LABELLED_KWARG_DATAFLOW_GRAPH_OUTPUT_LABEL_MAP_H

#include "utils/containers/generate_map.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_all_kwarg_dataflow_outputs.h"
#include "utils/graph/labelled_kwarg_dataflow_graph/labelled_kwarg_dataflow_graph_view.h"

namespace FlexFlow {

template <typename NodeLabel, typename OutputLabel, typename SlotName>
std::unordered_map<KwargDataflowOutput<SlotName>, OutputLabel>
    get_labelled_kwarg_dataflow_graph_output_label_map(
        LabelledKwargDataflowGraphView<NodeLabel, OutputLabel, SlotName> const
            &g) {
  return generate_map(
      get_all_kwarg_dataflow_outputs(g),
      [&](KwargDataflowOutput<SlotName> const &o) -> OutputLabel {
        return g.at(o);
      });
}

} // namespace FlexFlow

#endif
