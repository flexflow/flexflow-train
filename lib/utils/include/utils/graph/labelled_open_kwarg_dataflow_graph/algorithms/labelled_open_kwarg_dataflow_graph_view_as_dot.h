#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_VIEW_AS_DOT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_LABELLED_OPEN_KWARG_DATAFLOW_GRAPH_VIEW_AS_DOT_H

#include "utils/graph/labelled_open_kwarg_dataflow_graph/labelled_open_kwarg_dataflow_graph_view.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/open_kwarg_dataflow_graph_as_dot.h"

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
    std::function<nlohmann::json(NodeLabel const &)> const &render_node_label,
    std::function<nlohmann::json(ValueLabel const &)> const &render_value_label,
    std::function<nlohmann::json(SlotName const &)> const &render_slot_name,
    std::function<std::vector<SlotName>(
        std::unordered_set<SlotName> const &)> const &order_slots) {
  std::function<nlohmann::json(Node const &)> render_node =
      [&](Node const &n) -> nlohmann::json {
    return render_node_label(g.at(n));
  };

  std::function<nlohmann::json(
      OpenKwargDataflowValue<GraphInputName, SlotName> const &)>
      render_value =
          [&](OpenKwargDataflowValue<GraphInputName, SlotName> const &v)
      -> nlohmann::json { return render_value_label(g.at(v)); };

  return open_kwarg_dataflow_graph_as_dot(
      static_cast<OpenKwargDataflowGraphView<GraphInputName, SlotName>>(g),
      render_node,
      render_value,
      render_slot_name,
      order_slots);
}

} // namespace FlexFlow

#endif
