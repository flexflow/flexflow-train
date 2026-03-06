#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_KWARG_DATAFLOW_GRAPH_JSON_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_LABELLED_KWARG_DATAFLOW_GRAPH_JSON_H

#include "utils/graph/instances/unordered_set_labelled_open_kwarg_dataflow_graph.h"
#include "utils/graph/labelled_kwarg_dataflow_graph/algorithms/view_as_labelled_open_kwarg_dataflow_graph.h"
#include "utils/graph/labelled_kwarg_dataflow_graph/labelled_kwarg_dataflow_graph.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/algorithms/get_labelled_open_kwarg_dataflow_graph_data.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/algorithms/labelled_open_kwarg_dataflow_graph_data.dtg.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/algorithms/view_from_labelled_open_kwarg_dataflow_graph_data.h"
#include "utils/json/check_is_json_deserializable.h"
#include "utils/json/check_is_json_serializable.h"
#include "utils/json/monostate.h"
#include <nlohmann/json.hpp>
#include <variant>

namespace nlohmann {

template <typename NodeLabel, typename OutputLabel, typename SlotName>
struct adl_serializer<
    ::FlexFlow::LabelledKwargDataflowGraph<NodeLabel, OutputLabel, SlotName>> {
  static ::FlexFlow::
      LabelledKwargDataflowGraph<NodeLabel, OutputLabel, SlotName>
      from_json(json const &j) {
    CHECK_IS_JSON_DESERIALIZABLE(
        ::FlexFlow::LabelledOpenKwargDataflowGraphData<NodeLabel,
                                                       OutputLabel,
                                                       std::monostate,
                                                       SlotName>);

    auto data = j.template get<
        ::FlexFlow::LabelledOpenKwargDataflowGraphData<NodeLabel,
                                                       OutputLabel,
                                                       std::monostate,
                                                       SlotName>>();
    ::FlexFlow::LabelledOpenKwargDataflowGraphView<NodeLabel,
                                                   OutputLabel,
                                                   std::monostate,
                                                   SlotName>
        open_view =
            ::FlexFlow::view_from_labelled_open_kwarg_dataflow_graph_data(data);
    return ::FlexFlow::
        LabelledKwargDataflowGraph<NodeLabel, OutputLabel, SlotName>::
            template create_copy_of<
                ::FlexFlow::UnorderedSetLabelledOpenKwargDataflowGraph<
                    NodeLabel,
                    OutputLabel,
                    std::monostate,
                    SlotName>>(open_view);
  }

  static void
      to_json(json &j,
              ::FlexFlow::LabelledKwargDataflowGraph<NodeLabel,
                                                     OutputLabel,
                                                     SlotName> const &g) {
    CHECK_IS_JSON_SERIALIZABLE(
        ::FlexFlow::LabelledOpenKwargDataflowGraphData<NodeLabel,
                                                       OutputLabel,
                                                       std::monostate,
                                                       SlotName>);

    ::FlexFlow::LabelledOpenKwargDataflowGraphView<NodeLabel,
                                                   OutputLabel,
                                                   std::monostate,
                                                   SlotName>
        open_view = ::FlexFlow::view_as_labelled_open_kwarg_dataflow_graph<
            NodeLabel,
            OutputLabel,
            std::monostate,
            SlotName>(g);
    j = ::FlexFlow::get_labelled_open_kwarg_dataflow_graph_data(open_view);
  }
};

} // namespace nlohmann

#endif
