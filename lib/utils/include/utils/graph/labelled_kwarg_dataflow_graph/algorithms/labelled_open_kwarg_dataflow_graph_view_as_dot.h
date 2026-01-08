#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_KWARG_DATAFLOW_GRAPH_ALGORITHMS_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_VIEW_AS_DOT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_KWARG_DATAFLOW_GRAPH_ALGORITHMS_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_VIEW_AS_DOT_H

#include "utils/graph/labelled_open_kwarg_dataflow_graph/labelled_open_kwarg_dataflow_graph_view.h"

namespace FlexFlow {

template <typename NodeLabel,
          typename ValueLabel,
          typename GraphInputName,
          typename SlotName>
std::string labelled_open_kwarg_dataflow_graph_view_as_dot(
    LabelledOpenKwargDataflowGraphView<NodeLabel,
                                       ValueLabel,
                                       GraphInputName,
                                       SlotName> const &g,
    std::function<std::string(NodeLabel const &)> const &,
    std::function<std::string(ValueLabel const &)> const &) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow

#endif
