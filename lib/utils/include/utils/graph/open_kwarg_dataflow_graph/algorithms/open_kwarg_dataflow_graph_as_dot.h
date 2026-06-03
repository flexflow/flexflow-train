#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_OPEN_KWARG_DATAFLOW_GRAPH_AS_DOT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_OPEN_KWARG_DATAFLOW_GRAPH_AS_DOT_H

#include "utils/containers/filtrans.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/kwarg_dataflow_graph_as_dot.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/open_kwarg_dataflow_graph_as_dot.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/view_as_closed_kwarg_dataflow_graph_by_materializing_inputs.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_graph_view.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_value.dtg.h"

namespace FlexFlow {

template <typename GraphInputName, typename SlotName>
std::string open_kwarg_dataflow_graph_as_dot(
    OpenKwargDataflowGraphView<GraphInputName, SlotName> const &g) {
  std::function<nlohmann::json(Node const &)> render_node = [](Node const &n) {
    nlohmann::json j = fmt::to_string(n);
    return j;
  };

  std::function<nlohmann::json(
      OpenKwargDataflowValue<GraphInputName, SlotName> const &)>
      render_value =
          [](OpenKwargDataflowValue<GraphInputName, SlotName> const &v) {
            nlohmann::json j = fmt::to_string(v);
            return j;
          };

  std::function<nlohmann::json(SlotName const &)> render_slot_name =
      [](SlotName const &s) {
        nlohmann::json j = fmt::to_string(s);
        return j;
      };

  std::function<std::vector<SlotName>(std::unordered_set<SlotName> const &)>
      order_slots = [](std::unordered_set<SlotName> const &unordered) {
        return sorted(unordered);
      };

  return open_kwarg_dataflow_graph_as_dot(
      g, render_node, render_value, render_slot_name, order_slots);
}

template <typename GraphInputName, typename SlotName>
std::string open_kwarg_dataflow_graph_as_dot(
    OpenKwargDataflowGraphView<GraphInputName, SlotName> const &g,
    std::function<nlohmann::json(Node const &)> const &render_node,
    std::function<nlohmann::json(
        OpenKwargDataflowValue<GraphInputName, SlotName> const &)> const
        &render_value,
    std::function<nlohmann::json(SlotName const &)> const &render_slot_name,
    std::function<std::vector<SlotName>(
        std::unordered_set<SlotName> const &)> const &order_slots) {
  std::pair<KwargDataflowGraphView<std::optional<SlotName>>,
            bidict<KwargDataflowGraphInput<GraphInputName>, Node>>
      closed_g_and_mapping =
          view_as_closed_kwarg_dataflow_graph_by_materializing_inputs(g);

  KwargDataflowGraphView<std::optional<SlotName>> closed_g =
      closed_g_and_mapping.first;
  bidict<KwargDataflowGraphInput<GraphInputName>, Node> closed_mapping =
      closed_g_and_mapping.second;

  std::function<nlohmann::json(Node const &)> closed_render_node =
      [&](Node const &n) -> nlohmann::json {
    if (closed_mapping.contains_r(n)) {
      return render_value(OpenKwargDataflowValue<GraphInputName, SlotName>{
          closed_mapping.at_r(n)});
    } else {
      return render_node(n);
    }
  };

  std::function<nlohmann::json(
      KwargDataflowOutput<std::optional<SlotName>> const &)>
      closed_render_value =
          [&](KwargDataflowOutput<std::optional<SlotName>> const &o)
      -> nlohmann::json {
    if (closed_mapping.contains_r(o.node)) {
      ASSERT(!o.slot_name.has_value());

      nlohmann::json j = "graph_input";
      return j;
    } else {
      KwargDataflowOutput<SlotName> open_o = KwargDataflowOutput<SlotName>{
          /*node=*/o.node,
          /*slot_name=*/assert_unwrap(o.slot_name),
      };

      return render_value(
          OpenKwargDataflowValue<GraphInputName, SlotName>{open_o});
    }
  };

  std::function<nlohmann::json(std::optional<SlotName> const &)>
      closed_render_slot_name =
          [&](std::optional<SlotName> const &s) -> nlohmann::json {
    if (s.has_value()) {
      return render_slot_name(s.value());
    } else {
      nlohmann::json j = "graph_input";
      return j;
    }
  };

  std::function<std::vector<std::optional<SlotName>>(
      std::unordered_set<std::optional<SlotName>> const &)>
      closed_order_slots =
          [&](std::unordered_set<std::optional<SlotName>> const &unsorted)
      -> std::vector<std::optional<SlotName>> {
    std::unordered_set<SlotName> not_nullopt = filtrans(
        unsorted,
        [](std::optional<SlotName> const &s) -> std::optional<SlotName> {
          return s;
        });

    std::vector<SlotName> sorted_not_nullopt = order_slots(not_nullopt);

    std::vector<std::optional<SlotName>> result = transform(
        sorted_not_nullopt,
        [](SlotName const &s) -> std::optional<SlotName> { return s; });

    if (contains(unsorted, std::nullopt)) {
      result.push_back(std::nullopt);
    }

    return result;
  };

  return kwarg_dataflow_graph_as_dot(
      /*g=*/closed_g,
      /*render_node=*/closed_render_node,
      /*render_value=*/closed_render_value,
      /*render_slot_name=*/closed_render_slot_name,
      /*order_slots=*/closed_order_slots);
}

} // namespace FlexFlow

#endif
