#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_KWARG_DATAFLOW_GRAPH_AS_DOT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_KWARG_DATAFLOW_GRAPH_AS_DOT_H

#include "utils/graph/dataflow_graph/algorithms/dataflow_graph_as_dot.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/dataflow_graph_from_kwarg_dataflow_graph.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_incoming_slots_for_node.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_outgoing_slots_for_node.h"
#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_graph_view.h"

namespace FlexFlow {

template <typename SlotName>
std::string kwarg_dataflow_graph_as_dot(
    KwargDataflowGraphView<SlotName> const &g,
    std::function<nlohmann::json(Node const &)> const &render_node,
    std::function<nlohmann::json(KwargDataflowOutput<SlotName> const &)> const
        &render_value,
    std::function<nlohmann::json(SlotName const &)> const &render_slot_name,
    std::function<std::vector<SlotName>(
        std::unordered_set<SlotName> const &)> const &order_slots) {
  std::function<nlohmann::json(DataflowInput const &)> get_input_label =
      [&](DataflowInput const &i) -> nlohmann::json {
    std::vector<SlotName> slot_ordering =
        order_slots(get_incoming_slots_for_node(g, i.node));

    SlotName slot_name = slot_ordering.at(i.idx.unwrap_nonnegative());

    return render_slot_name(slot_name);
  };

  std::function<nlohmann::json(DataflowOutput const &)> get_output_label =
      [&](DataflowOutput const &o) -> nlohmann::json {
    std::vector<SlotName> slot_ordering =
        order_slots(get_outgoing_slots_for_node(g, o.node));

    SlotName slot_name = slot_ordering.at(o.idx.unwrap_nonnegative());

    nlohmann::json result;

    result["slot"] = render_slot_name(slot_name);

    KwargDataflowOutput<SlotName> kwarg_o = KwargDataflowOutput<SlotName>{
        /*node=*/o.node,
        /*slot_name=*/slot_name,
    };
    result["value"] = render_value(kwarg_o);

    return result;
  };

  return dataflow_graph_as_dot(
      dataflow_graph_from_kwarg_dataflow_graph(g, order_slots),
      render_node,
      /*get_input_label=*/get_input_label,
      /*get_output_label=*/get_output_label);
}

} // namespace FlexFlow

#endif
